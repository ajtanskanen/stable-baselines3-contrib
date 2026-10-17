"""
Kohta kohdalta -testiohjelma SB2 ACKTR -> SB3/PyTorch -portille (audit v31).

Testattava koodi ladataan sb3-contribista. Oletuksena kokeillaan järjestyksessä
sb3_contrib.acktr.acktr -> sb3_contrib.acktr -> sb3_contrib, ja vasta viimeisenä
irrallista acktr_kfac-moduulia. Ympäristömuuttuja ACKTR_MODULE ohittaa haun.

Ajo:
    pip install stable-baselines3 gymnasium scipy pytest torch
    pip install -e /polku/omaan/stable-baselines3-contrib
    pytest -v test_acktr_kfac.py

    # tai suoraan lähdepuusta ilman asennusta:
    PYTHONPATH=/polku/omaan/stable-baselines3-contrib pytest -v test_acktr_kfac.py

    # tai osoita moduuli itse:
    ACKTR_MODULE=sb3_contrib.acktr.acktr pytest -v test_acktr_kfac.py

Testit on ryhmitelty osioihin, jotka vastaavat lähdekoodin rakennetta:

    0  Moduulin lataus sb3-contribista ja paketointi

    1  make_sb2_lr_schedule            (SB2 Scheduler -kaavat)
    2  MPS-kyvykkyysprobet
    3  SB2KFACOptimizer: konstruktorin validointi
    4  Forward-hookien elinkaari
    5  Fisher-keruun tilakone
    6  Linear-kovarianssit (SB2 T1-skaalaus, yksi Fisher-pari per kerros)
    7  Conv2d-kovarianssit
    8  Faktorimuodot ja statistiikan akkumulaatio (stats_accum_iter vs EMA)
    9  Ominaisarvojen leikkaus ja _safe_symmetric_eigh (torch/scipy/float64)
    10 _update_inverses: atomisuus, portti, kylmä restart
    11 _register_refresh_failure: toistuva epäonnistuminen on fataali
    12 Luonnollinen gradientti (eksakti K-FAC-ratkaisu)
    13 step(): kylmä SGD, lämmin päivitys, KL-skaalaus, refresh-väli
    14 Checkpointointi (state_dict / load_state_dict)
    15 ACKTR + SB3: rakennus, action spacet, runkoasettelu
    16 ACKTR + SB3: oikea koulutus (MLP ja CNN), lokit, gradienttileikkaus

Suite seuraa v31:n rajapintaa. v31:stä poistettujen valintojen testit
(sb2_reference_mode, use_kfac, stable blend, eigen_backend, save_eigen_state,
optimizer_steps_per_rollout, monen Fisher-parin dedup ym.) on poistettu.
Aiemmat xfail-löydökset (test_13_4b refresh-väli, test_7_2c conv-dedup) on
korjattu tai niiden kohde poistettu v31:ssä.
"""

from __future__ import annotations

import copy
import importlib
import math
import os
import sys
import warnings

import numpy as np
import pytest
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces


# ======================================================================
# Testattavan koodin lataus sb3-contribista
# ======================================================================

MODULE_CANDIDATES = (
    "sb3_contrib.acktr.acktr",
    "sb3_contrib.acktr",
    "sb3_contrib",
    "acktr_kfac",
)

# Jos ACKTR ja K-FAC-optimizer on jaettu useaan tiedostoon, symboleja etsitään
# myös näistä saman alipaketin moduuleista.
SIBLING_MODULES = ("acktr", "kfac", "kfac_optimizer", "policies")


def _import_entry_point():
    """Etsi moduuli, joka vie ACKTR-luokan ulos."""
    override = os.environ.get("ACKTR_MODULE")
    names = [override] if override else list(MODULE_CANDIDATES)
    problems = []
    for name in names:
        try:
            module = importlib.import_module(name)
        except Exception as exc:  # ImportError, mutta myös rikkinäinen __init__
            problems.append(f"  {name}: {type(exc).__name__}: {exc}")
            continue
        if getattr(module, "ACKTR", None) is None:
            problems.append(f"  {name}: moduulissa ei ole ACKTR-luokkaa")
            continue
        return module
    raise ImportError(
        "ACKTR:ää ei löytynyt. Asenna paikallinen sb3-contrib "
        "(pip install -e /polku/stable-baselines3-contrib), lisää se "
        "PYTHONPATH:iin tai osoita moduuli ACKTR_MODULE-muuttujalla.\n"
        "Kokeillut sijainnit:\n" + "\n".join(problems)
    )


_ENTRY = _import_entry_point()
ACKTR = _ENTRY.ACKTR
# Monkeypatch-testit muokkaavat sen moduulin nimiavaruutta, jossa symbolit on
# MÄÄRITELTY (M.F, M.th, M.mps_*), eivät sitä joka vain re-exporttaa ne.
# Siksi seurataan __module__-viittausta pakettien __init__-tiedostojen läpi.
_DEFINING = sys.modules.get(ACKTR.__module__, _ENTRY)
_PACKAGE = ACKTR.__module__.rpartition(".")[0]


def _resolve_symbol(name: str):
    for module in (_DEFINING, _ENTRY):
        value = getattr(module, name, None)
        if value is not None:
            return value
    for sibling in SIBLING_MODULES:
        if not _PACKAGE:
            break
        try:
            module = importlib.import_module(f"{_PACKAGE}.{sibling}")
        except Exception:
            continue
        value = getattr(module, name, None)
        if value is not None:
            return value
    raise ImportError(
        f"Symbolia {name!r} ei löytynyt moduulista {ACKTR.__module__!r} "
        "eikä sen sisarmoduuleista."
    )


SB2KFACOptimizer = _resolve_symbol("SB2KFACOptimizer")
KFACModuleState = _resolve_symbol("KFACModuleState")
make_sb2_lr_schedule = _resolve_symbol("make_sb2_lr_schedule")

# M = moduuli jota monkeypatchataan (K-FAC-apurien nimiavaruus).
M = sys.modules[SB2KFACOptimizer.__module__]
ACKTR_DEFINING_MODULE = sys.modules[ACKTR.__module__]
FROM_SB3_CONTRIB = ACKTR.__module__.split(".")[0] == "sb3_contrib"


# ======================================================================
# 0. Moduulin lataus ja paketointi
# ======================================================================


def test_0_1_loaded_symbols_have_the_expected_types():
    from stable_baselines3 import A2C

    assert issubclass(ACKTR, A2C)
    assert issubclass(SB2KFACOptimizer, th.optim.Optimizer)
    assert callable(make_sb2_lr_schedule)
    assert KFACModuleState().stats_count == 0
    assert hasattr(M, "mps_is_available")
    assert isinstance(getattr(M, "SB2_AUDIT_VERSION", ""), str)


@pytest.mark.skipif(
    not FROM_SB3_CONTRIB,
    reason=f"ACKTR ladattiin moduulista {ACKTR.__module__}, ei sb3-contribista",
)
def test_0_2_acktr_is_exported_from_the_sb3_contrib_top_level():
    """sb3-contribin konventio: algoritmi on saatavilla paketin juuresta."""
    import sb3_contrib

    assert getattr(sb3_contrib, "ACKTR", None) is ACKTR, (
        "Lisää sb3_contrib/__init__.py:hyn: from sb3_contrib.acktr import ACKTR"
    )
    exported = getattr(sb3_contrib, "__all__", None)
    if exported is not None:
        assert "ACKTR" in exported, (
            "Lisää \"ACKTR\" myös sb3_contrib/__init__.py:n __all__-listaan."
        )


@pytest.mark.skipif(not FROM_SB3_CONTRIB, reason="Ei sb3-contrib-asennus")
def test_0_3_acktr_subpackage_layout():
    package = importlib.import_module(_PACKAGE) if _PACKAGE else _ENTRY
    assert getattr(package, "ACKTR", None) is ACKTR


# ======================================================================
# Apurit
# ======================================================================


class TinyNet(nn.Module):
    """Kaksi Linear-kerrosta; pienin malli jossa on useampi K-FAC-lohko."""

    def __init__(self, bias: bool = True, in_dim: int = 4, hidden: int = 3, out_dim: int = 2):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden, bias=bias)
        self.fc2 = nn.Linear(hidden, out_dim, bias=bias)

    def forward(self, x):
        return self.fc2(th.tanh(self.fc1(x)))


class TinyConvNet(nn.Module):
    def __init__(self, bias: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(2, 3, kernel_size=2, bias=bias)

    def forward(self, x):
        return self.conv(x)


def make_opt(model: nn.Module, **kwargs) -> SB2KFACOptimizer:
    """Optimizer testiystävällisillä oletuksilla (nopea kypsyminen)."""
    params = dict(
        lr=0.1,
        damping=0.01,
        stats_decay=0.5,
        kfac_update=1,
        kfac_clip=0.001,
        momentum=0.0,
        cold_iter=0,
        stats_accum_iter=1,
    )
    params.update(kwargs)
    return SB2KFACOptimizer(model, **params)


def feed_pairs(opt: SB2KFACOptimizer, module: nn.Module, pairs) -> None:
    """Syötä (x, g) -parit suoraan ohi hookien ja commitoi yksi stats-askel."""
    opt.begin_fisher_stats_collection()
    opt._pending[module].extend(pairs)
    opt.complete_fisher_stats_collection()


def sample_through_hooks(opt: SB2KFACOptimizer, model: nn.Module, x: th.Tensor) -> None:
    """Kerää yksi Fisher-näyte oikean forward/backward-polun kautta."""
    opt.begin_fisher_stats_collection()
    out = model(x)
    out.sum().backward()
    opt.complete_fisher_stats_collection()
    model.zero_grad(set_to_none=True)


def spd(n: int, seed: int = 0, dtype=th.float32) -> th.Tensor:
    """Determinististä hyvin ehdollistettua symmetristä positiividefiniittiä."""
    gen = th.Generator().manual_seed(seed)
    base = th.randn(n, n, generator=gen, dtype=th.float64)
    matrix = base @ base.t() / n + th.eye(n, dtype=th.float64) * float(n)
    return matrix.to(dtype)


def set_spectral_state(
    opt: SB2KFACOptimizer, module: nn.Module, A: th.Tensor, G: th.Tensor, mature: bool = True
) -> None:
    """Aseta A/G ja niiden hajotelmat käsin (ohittaa Fisher-keruun)."""
    state = opt.module_states[module]
    state.A = A.clone()
    state.G = G.clone()
    a_vals, a_vecs = th.linalg.eigh(A.double())
    g_vals, g_vecs = th.linalg.eigh(G.double())
    state.A_eigvals = a_vals.to(A.dtype)
    state.A_eigvecs = a_vecs.to(A.dtype)
    state.G_eigvals = g_vals.to(G.dtype)
    state.G_eigvecs = g_vecs.to(G.dtype)
    state.stats_count = opt.stats_accum_iter if mature else 0


# ======================================================================
# 1. make_sb2_lr_schedule -- SB2 Scheduler -kaavojen tarkka vastaavuus
# ======================================================================


def sb2_reference_schedule(name: str, elapsed: float) -> float:
    """Riippumaton toteutus SB2:n legacy Scheduler-kaavoista.

    SB2 antaa schedule-funktiolle KULUNEEN osuuden (step / n_values),
    SB3 antaa jäljellä olevan. Portin pitää tehdä muunnos.
    """
    if name == "constant":
        return 1.0
    if name == "linear":
        return 1.0 - elapsed
    if name == "double_linear_con":
        doubled = elapsed * 2.0
        return 0.125 if (1.0 - doubled) < 0.125 else (1.0 - doubled)
    if name == "middle_drop":
        return 0.75 * 0.1 if (1.0 - elapsed) < 0.75 else (1.0 - elapsed)
    if name == "double_middle_drop":
        if (1.0 - elapsed) < 0.75:
            if (1.0 - elapsed) < 0.25:
                return 0.25 * 0.5
            return 0.75 * 0.1
        return 1.0 - elapsed
    raise AssertionError(name)


@pytest.mark.parametrize(
    "name",
    ["constant", "linear", "double_linear_con", "middle_drop", "double_middle_drop"],
)
def test_1_1_schedules_match_sb2_formulas(name):
    initial = 0.25
    schedule = make_sb2_lr_schedule(initial, name)
    for elapsed in [0.0, 0.1, 0.24, 0.25, 0.26, 0.5, 0.74, 0.75, 0.76, 0.9, 1.0]:
        progress_remaining = 1.0 - elapsed
        expected = initial * sb2_reference_schedule(name, elapsed)
        assert schedule(progress_remaining) == pytest.approx(expected, rel=1e-12, abs=1e-12), (
            f"{name} @ elapsed={elapsed}"
        )


def test_1_2_linear_endpoints():
    schedule = make_sb2_lr_schedule(0.25, "linear")
    assert schedule(1.0) == pytest.approx(0.25)
    assert schedule(0.0) == pytest.approx(0.0)
    assert schedule(0.5) == pytest.approx(0.125)


def test_1_3_double_middle_drop_step_shape():
    # SB2:ssa loppupää (0.125) on ISOMPI kuin keskikaista (0.075). Tämä on
    # alkuperäisen kaavan ominaisuus, ei portin bugi -- lukitaan se testiin.
    schedule = make_sb2_lr_schedule(1.0, "double_middle_drop")
    assert schedule(0.80) == pytest.approx(0.80)
    assert schedule(0.50) == pytest.approx(0.075)
    assert schedule(0.10) == pytest.approx(0.125)
    assert schedule(0.75) == pytest.approx(0.75)   # raja kuuluu ylempään haaraan
    assert schedule(0.25) == pytest.approx(0.075)  # raja kuuluu keskikaistaan


def test_1_4_callable_is_passed_through_unchanged():
    def custom(progress_remaining: float) -> float:
        return 0.5 * progress_remaining

    assert make_sb2_lr_schedule(custom, "linear") is custom


def test_1_5_unknown_schedule_raises_on_construction():
    with pytest.raises(ValueError, match="Unknown lr_schedule"):
        make_sb2_lr_schedule(0.25, "ei-olemassa")




def test_1_6_schedule_name_is_case_insensitive():
    assert make_sb2_lr_schedule(0.25, "LINEAR")(0.5) == pytest.approx(0.125)


# ======================================================================
# 2. MPS-probet (CPU-koneella nämä testaavat turvallisen "ei tuettu" -polun)
# ======================================================================


def test_2_1_mps_is_available_never_raises(monkeypatch):
    class Broken:
        @staticmethod
        def is_available():
            raise RuntimeError("rikki")

    monkeypatch.setattr(M.th.backends, "mps", Broken(), raising=False)
    assert M.mps_is_available() is False


def test_2_2_probe_returns_false_without_mps(monkeypatch):
    monkeypatch.setattr(M, "mps_is_available", lambda: False)
    assert M.mps_supports_reliable_eigh.__wrapped__() is False




def test_2_4_probe_swallows_exceptions(monkeypatch):
    monkeypatch.setattr(M, "mps_is_available", lambda: True)

    def failing(*args, **kwargs):
        raise NotImplementedError("ei Metal-ydintä")

    monkeypatch.setattr(M.th.linalg, "eigh", failing)
    assert M.mps_supports_reliable_eigh.__wrapped__() is False




def test_2_5_capability_probe_is_cached():
    assert M.mps_supports_reliable_eigh.cache_info().maxsize == 1
    first = M.mps_supports_reliable_eigh()
    assert M.mps_supports_reliable_eigh() is first
    assert M.mps_supports_reliable_eigh.cache_info().currsize == 1




# ======================================================================
# 3. SB2KFACOptimizer -- konstruktorin validointi
# ======================================================================


def test_3_1_requires_trainable_parameters():
    model = TinyNet()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    with pytest.raises(ValueError, match="no trainable parameters"):
        make_opt(model)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (dict(stats_decay=1.0), "stats_decay must be in"),
        (dict(stats_decay=-0.1), "stats_decay must be in"),
        (dict(momentum=1.0), "momentum must be in"),
        (dict(damping=0.0), "damping must be positive"),
        (dict(kfac_clip=0.0), "kfac_clip must be positive"),
        (dict(eps=0.0), "eps must be positive"),
    ],
)
def test_3_2_parameter_validation(kwargs, message):
    with pytest.raises(ValueError, match=message):
        make_opt(TinyNet(), **kwargs)


def test_3_3_counters_start_from_sb2_defaults():
    opt = make_opt(TinyNet(), kfac_update=7, cold_iter=3, stats_accum_iter=5)
    assert opt.steps == 0
    assert opt.stats_steps == 0
    assert opt.factor_steps == 0
    assert opt.eigh_failures == 0
    assert opt.consecutive_refresh_failures == 0
    assert opt.last_eigh_backend == "none"
    # Refresh-väli mitataan edellisestä ONNISTUNEESTA päivityksestä, joten
    # laskuri alustetaan täyteen: ensimmäinen kypsä lämmin askel päivittää heti.
    assert opt._steps_since_refresh == opt.kfac_update == 7
    assert opt.cold_iter == 3
    assert opt.stats_accum_iter == 5
    assert opt.preconditioner_is_active() is False


def test_3_4_stats_accum_iter_is_independent_of_cold_iter():
    opt = make_opt(TinyNet(), cold_iter=10, stats_accum_iter=60)
    assert opt.cold_iter == 10
    assert opt.stats_accum_iter == 60


def test_3_5_tied_weights_are_rejected():
    model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
    model[1].weight = model[0].weight
    with pytest.raises(NotImplementedError, match="tied weights"):
        make_opt(model)




def test_3_6_unsupported_conv_configurations():
    with pytest.raises(NotImplementedError, match="Grouped/depthwise"):
        make_opt(nn.Conv2d(4, 4, 3, groups=2))
    with pytest.raises(NotImplementedError, match="String Conv2d padding"):
        make_opt(nn.Conv2d(2, 3, 3, padding="same"))
    with pytest.raises(NotImplementedError, match="only zero padding"):
        make_opt(nn.Conv2d(2, 3, 3, padding=1, padding_mode="reflect"))


def test_3_7_frozen_weight_module_is_not_a_kfac_block():
    model = TinyNet()
    model.fc1.weight.requires_grad_(False)
    opt = make_opt(model)
    assert model.fc1 not in opt.module_states
    assert opt.modules == [model.fc2]
    # fc1.bias on yhä treenattava ja hoidetaan euklidisella fallback-polulla.
    assert model.fc1.bias.requires_grad


def test_3_8_all_kfac_modules_are_registered():
    model = TinyNet()
    opt = make_opt(model)
    assert opt.modules == [model.fc1, model.fc2]
    assert len(opt.handles) == 2
    assert set(opt._pending) == {model.fc1, model.fc2}


# ======================================================================
# 4. Forward-hookien elinkaari
# ======================================================================


def test_4_1_remove_hooks_detaches_and_clears():
    model = TinyNet()
    opt = make_opt(model)
    opt.remove_hooks()
    assert opt.handles == []
    assert opt.collecting_fisher_stats is False

    opt.begin_fisher_stats_collection()
    model(th.randn(5, 4)).sum().backward()
    with pytest.raises(RuntimeError, match="No Linear/Conv2d Fisher factors"):
        opt.complete_fisher_stats_collection()
    assert opt.collecting_fisher_stats is False


def test_4_3_hooks_capture_nothing_when_not_collecting():
    model = TinyNet()
    opt = make_opt(model)
    model(th.randn(5, 4)).sum().backward()
    assert all(len(pairs) == 0 for pairs in opt._pending.values())


def test_4_4_hook_respects_flag_at_backward_time():
    model = TinyNet()
    opt = make_opt(model)
    opt.begin_fisher_stats_collection()
    out = model(th.randn(5, 4))
    opt.discard_fisher_stats_collection()
    out.sum().backward()
    assert all(len(pairs) == 0 for pairs in opt._pending.values())


def test_4_5_no_grad_forward_is_ignored():
    model = TinyNet()
    opt = make_opt(model)
    opt.begin_fisher_stats_collection()
    with th.no_grad():
        model(th.randn(5, 4))
    assert all(len(pairs) == 0 for pairs in opt._pending.values())
    opt.discard_fisher_stats_collection()


# ======================================================================
# 5. Fisher-keruun tilakone
# ======================================================================


def test_5_1_double_begin_is_an_error():
    opt = make_opt(TinyNet())
    opt.begin_fisher_stats_collection()
    with pytest.raises(RuntimeError, match="already active"):
        opt.begin_fisher_stats_collection()


def test_5_2_complete_without_begin_is_an_error():
    opt = make_opt(TinyNet())
    with pytest.raises(RuntimeError, match="not active"):
        opt.complete_fisher_stats_collection()


def test_5_4_begin_clears_stale_pairs():
    model = TinyNet()
    opt = make_opt(model)
    opt._pending[model.fc1].append((th.randn(2, 4), th.randn(2, 3)))
    opt.begin_fisher_stats_collection()
    assert opt._pending[model.fc1] == []
    opt.discard_fisher_stats_collection()


def test_5_5_failure_inside_complete_releases_the_collection(monkeypatch):
    model = TinyNet()
    opt = make_opt(model)

    def boom(*args, **kwargs):
        raise RuntimeError("kovarianssi kaatui")

    monkeypatch.setattr(opt, "_linear_factors", boom)
    opt.begin_fisher_stats_collection()
    opt._pending[model.fc1].append((th.randn(2, 4), th.randn(2, 3)))
    with pytest.raises(RuntimeError, match="kovarianssi kaatui"):
        opt.complete_fisher_stats_collection()
    assert opt.collecting_fisher_stats is False
    assert opt._pending[model.fc1] == []
    assert opt.stats_steps == 0
    assert opt.module_states[model.fc1].A is None




def test_5_6_stats_step_counter_increments_once_per_commit():
    model = TinyNet()
    opt = make_opt(model)
    for expected in (1, 2, 3):
        sample_through_hooks(opt, model, th.randn(6, 4))
        assert opt.stats_steps == expected


# ======================================================================
# 6. Linear-kovarianssit
# ======================================================================


def test_6_1_linear_factors_without_bias():
    module = nn.Linear(4, 3, bias=False)
    opt = make_opt(module)
    x = th.randn(7, 4)
    g = th.randn(7, 3)
    A, G = opt._linear_factors(module, x, g)
    assert th.allclose(A, x.t() @ x / 7.0, atol=1e-6)
    # SB2 T1: kertoo takaperinfaktorin näytemäärällä ennen kovarianssia.
    assert th.allclose(G, (g.t() @ g) * 7.0, atol=1e-4)




def test_6_2_linear_factors_with_bias_append_a_ones_column():
    module = nn.Linear(4, 3, bias=True)
    opt = make_opt(module)
    x = th.randn(7, 4)
    g = th.randn(7, 3)
    A, G = opt._linear_factors(module, x, g)
    augmented = th.cat([x, th.ones(7, 1)], dim=1)
    assert A.shape == (5, 5)
    assert th.allclose(A, augmented.t() @ augmented / 7.0, atol=1e-6)
    assert A[-1, -1] == pytest.approx(1.0, abs=1e-6)


def test_6_3_several_backprops_through_one_layer_are_rejected():
    """v31 olettaa YHDEN summatun Fisher-backwardin per kerros (kuten SB2)."""
    module = nn.Linear(4, 3, bias=False)
    opt = make_opt(module)
    x = th.randn(7, 4)
    opt.begin_fisher_stats_collection()
    opt._pending[module].extend([(x, th.randn(7, 3)), (x, th.randn(7, 3))])
    with pytest.raises(NotImplementedError, match="one summed Fisher backward"):
        opt.complete_fisher_stats_collection()
    assert opt.collecting_fisher_stats is False
    assert opt.stats_steps == 0
    assert opt.module_states[module].A is None




def test_6_5_time_major_inputs_are_flattened():
    module = nn.Linear(4, 3, bias=False)
    opt = make_opt(module)
    x = th.randn(5, 6, 4)
    g = th.randn(5, 6, 3)
    A, G = opt._linear_factors(module, x, g)
    flat_x = x.reshape(-1, 4)
    assert th.allclose(A, flat_x.t() @ flat_x / 30.0, atol=1e-6)
    assert G.shape == (3, 3)




def test_6_6_row_mismatch_is_detected():
    module = nn.Linear(4, 3, bias=False)
    opt = make_opt(module)
    with pytest.raises(RuntimeError, match="row mismatch"):
        opt._linear_factors(module, th.randn(7, 4), th.randn(5, 3))




def test_6_7_empty_sample_is_an_error():
    module = nn.Linear(4, 3, bias=False)
    opt = make_opt(module)
    with pytest.raises(RuntimeError, match="Empty Linear Fisher sample"):
        opt._linear_factors(module, th.randn(0, 4), th.randn(0, 3))




# ======================================================================
# 7. Conv2d-kovarianssit
# ======================================================================


def conv_reference_factors(module, x, g_list):
    """Riippumaton referenssitoteutus SB2:n conv-faktoreille."""
    patches = F.unfold(
        x,
        kernel_size=module.kernel_size,
        dilation=module.dilation,
        padding=module.padding,
        stride=module.stride,
    )
    spatial = patches.shape[-1]
    rows = patches.transpose(1, 2).reshape(-1, patches.shape[1]) / float(spatial)
    if module.bias is not None and module.bias.requires_grad:
        rows = th.cat([rows, th.ones(rows.shape[0], 1) / float(spatial)], dim=1)
    A = rows.t() @ rows / float(x.shape[0])

    g_sum = None
    for g in g_list:
        g_rows = g.permute(0, 2, 3, 1).reshape(-1, g.shape[1])
        block = g_rows.t() @ g_rows
        g_sum = block if g_sum is None else g_sum + block
    total_batch = x.shape[0] * len(g_list)
    G = g_sum * float(total_batch * spatial)
    return A, G


def test_7_1_conv_factors_match_the_reference():
    model = TinyConvNet(bias=True)
    opt = make_opt(model)
    x = th.randn(4, 2, 5, 5)
    g = th.randn(4, 3, 4, 4)
    A, G = opt._conv2d_factors(model.conv, x, g)
    ref_A, ref_G = conv_reference_factors(model.conv, x, [g])
    assert A.shape == (2 * 2 * 2 + 1, 2 * 2 * 2 + 1)
    assert G.shape == (3, 3)
    assert th.allclose(A, ref_A, atol=1e-5)
    assert th.allclose(G, ref_G, rtol=1e-4, atol=1e-3)




def test_7_3_conv_requires_nchw():
    model = TinyConvNet()
    opt = make_opt(model)
    with pytest.raises(RuntimeError, match="expects NCHW"):
        opt._conv2d_factors(model.conv, th.randn(4, 2, 5), th.randn(4, 3, 4, 4))




def test_7_5_conv_spatial_mismatch_between_unfold_and_grad():
    model = TinyConvNet()
    opt = make_opt(model)
    with pytest.raises(RuntimeError, match="spatial mismatch"):
        opt._conv2d_factors(model.conv, th.randn(2, 2, 5, 5), th.randn(2, 3, 3, 3))




def test_7_6_conv_factors_through_the_real_hook_path():
    model = TinyConvNet(bias=True)
    opt = make_opt(model)
    x = th.randn(3, 2, 6, 6)
    sample_through_hooks(opt, model, x)
    state = opt.module_states[model.conv]
    assert state.A.shape == (9, 9)
    assert state.G.shape == (3, 3)
    assert state.stats_count == 1


# ======================================================================
# 8. Faktorimuodot ja statistiikan akkumulaatio
# ======================================================================


def test_8_1_expected_factor_shapes():
    opt = make_opt(TinyNet())
    linear_bias = nn.Linear(4, 3, bias=True)
    linear_plain = nn.Linear(4, 3, bias=False)
    conv = nn.Conv2d(2, 3, kernel_size=(2, 3), bias=True)
    assert opt._expected_factor_shapes(linear_bias) == ((5, 5), (3, 3))
    assert opt._expected_factor_shapes(linear_plain) == ((4, 4), (3, 3))
    assert opt._expected_factor_shapes(conv) == ((13, 13), (3, 3))
    with pytest.raises(TypeError, match="Unsupported K-FAC module"):
        opt._expected_factor_shapes(nn.ReLU())


def test_8_2_initial_accumulation_is_equally_weighted_then_ema():
    """SB2: ensimmäiset stats_accum_iter näytettä painolla 1/N, sitten EMA."""
    module = nn.Linear(3, 2, bias=False)
    opt = make_opt(module, stats_accum_iter=3, stats_decay=0.5)
    state = opt.module_states[module]

    x1 = th.randn(5, 3)
    g1 = th.randn(5, 2)
    A1 = x1.t() @ x1 / 5.0

    for k in (1, 2, 3):
        feed_pairs(opt, module, [(x1, g1)])
        assert state.stats_count == k
        assert th.allclose(state.A, A1 * (k / 3.0), atol=1e-6)

    mature_A = state.A.clone()
    x2 = th.randn(5, 3) * 3.0
    A2 = x2.t() @ x2 / 5.0
    feed_pairs(opt, module, [(x2, g1)])
    assert state.stats_count == 4
    assert th.allclose(state.A, mature_A * 0.5 + A2 * 0.5, atol=1e-5)


def test_8_3_shape_change_is_rejected_and_state_is_untouched():
    module = nn.Linear(3, 2, bias=False)
    opt = make_opt(module)
    poisoned = th.eye(9)
    opt.module_states[module].A = poisoned
    opt.module_states[module].G = th.eye(2)
    with pytest.raises(RuntimeError, match="A shape changed"):
        feed_pairs(opt, module, [(th.randn(4, 3), th.randn(4, 2))])
    assert opt.module_states[module].A is poisoned
    assert opt.stats_steps == 0


def test_8_4_commit_is_atomic_across_modules():
    model = TinyNet(bias=False)
    opt = make_opt(model, stats_accum_iter=2)
    # fc2 on "kypsä" mutta ilman kovariansseja -> arithmetiikka kaatuu.
    opt.module_states[model.fc2].stats_count = 2
    opt.begin_fisher_stats_collection()
    opt._pending[model.fc1].append((th.randn(4, 4), th.randn(4, 3)))
    opt._pending[model.fc2].append((th.randn(4, 3), th.randn(4, 2)))
    with pytest.raises(RuntimeError, match="Invalid mature K-FAC state"):
        opt.complete_fisher_stats_collection()
    assert opt.module_states[model.fc1].A is None
    assert opt.module_states[model.fc1].stats_count == 0
    assert opt.stats_steps == 0




def test_8_5_debug_finite_checks_gate_the_expensive_validation():
    module = nn.Linear(3, 2, bias=False)
    quiet = make_opt(module, debug_finite_checks=False)
    x = th.full((4, 3), float("inf"))
    g = th.randn(4, 2)
    feed_pairs(quiet, module, [(x, g)])  # ei nosta poikkeusta
    assert not th.isfinite(quiet.module_states[module].A).all()

    module2 = nn.Linear(3, 2, bias=False)
    loud = make_opt(module2, debug_finite_checks=True)
    with pytest.raises(FloatingPointError, match="Non-finite"):
        feed_pairs(loud, module2, [(th.full((4, 3), float("inf")), th.randn(4, 2))])


# ======================================================================
# 9. Ominaisarvojen leikkaus ja _safe_symmetric_eigh
# ======================================================================


def test_9_1_clamp_uses_a_relative_rank_tolerance():
    clamp = make_opt(TinyNet())._clamp_eigenvalues
    vals = th.tensor([1e-12, 1.0])
    assert clamp(vals).tolist() == [0.0, 1.0]
    # Pienimittakaavainen faktori: kiinteä 1e-6 olisi nollannut kaiken,
    # suhteellinen kynnys (max * n * eps) säilyttää molemmat.
    small = th.tensor([1e-7, 1e-5])
    assert th.all(clamp(small) > 0)
    assert th.allclose(clamp(th.tensor([0.5, 1.0])), th.tensor([0.5, 1.0]))




def test_9_2_clamp_degenerate_inputs():
    clamp = make_opt(TinyNet())._clamp_eigenvalues
    assert clamp(th.tensor([])).numel() == 0
    assert th.all(clamp(th.tensor([-2.0, -1.0])) == 0)
    assert th.all(clamp(th.tensor([float("inf"), 1.0])) == 0)




def test_9_3_eigh_shape_validation():
    module = nn.Linear(3, 2, bias=False)
    opt = make_opt(module)
    with pytest.raises(ValueError, match="must be square"):
        opt._safe_symmetric_eigh(th.randn(3, 4), module=module, factor_name="A")
    with pytest.raises(ValueError, match="is empty"):
        opt._safe_symmetric_eigh(th.zeros(0, 0), module=module, factor_name="A")


def test_9_4_eigh_rejects_non_finite_input_unconditionally():
    module = nn.Linear(3, 2, bias=False)
    opt = make_opt(module, debug_finite_checks=False)
    matrix = th.eye(3)
    matrix[0, 0] = float("nan")
    with pytest.raises(FloatingPointError, match="Non-finite values in K-FAC A factor"):
        opt._safe_symmetric_eigh(matrix, module=module, factor_name="A")


def test_9_5_eigh_reconstructs_the_matrix_on_cpu_via_scipy():
    module = nn.Linear(6, 2, bias=False)
    opt = make_opt(module)
    A = spd(6, seed=1)
    vals, vecs, backend = opt._safe_symmetric_eigh(A, module=module, factor_name="A")
    # Oletuksena kaikki CPU-faktorit kokoon katsomatta menevät SYEVR:lle.
    assert backend == "scipy-evr"
    assert vals.dtype == A.dtype and vecs.dtype == A.dtype
    rebuilt = (vecs * vals.unsqueeze(0)) @ vecs.t()
    assert th.allclose(rebuilt, A, atol=1e-3)


def test_9_6_eigh_symmetrises_the_input():
    module = nn.Linear(4, 2, bias=False)
    opt = make_opt(module)
    base = spd(4, seed=2)
    skewed = base.clone()
    skewed[0, 1] += 5.0  # rikkoo symmetrian
    vals, _, _ = opt._safe_symmetric_eigh(skewed, module=module, factor_name="A")
    expected = th.linalg.eigvalsh(0.5 * (skewed + skewed.t()))
    assert th.allclose(vals, expected, atol=1e-3)


def test_9_7_cpu_factors_are_routed_to_scipy_syevr(monkeypatch):
    module = nn.Linear(8, 2, bias=False)
    opt = make_opt(module)
    A = spd(8, seed=3)
    assert M._SCIPY_EIGH_THRESHOLD <= 1
    vals, vecs, backend = opt._safe_symmetric_eigh(A, module=module, factor_name="A")
    assert backend == "scipy-evr"
    assert th.allclose((vecs * vals.unsqueeze(0)) @ vecs.t(), A, atol=1e-3)

    monkeypatch.setattr(M, "_SCIPY_EIGH_THRESHOLD", 100)
    _, _, backend_small = opt._safe_symmetric_eigh(A, module=module, factor_name="A")
    assert backend_small == "torch-cpu"




def test_9_9_torch_failure_falls_back_to_scipy(monkeypatch):
    module = nn.Linear(4, 2, bias=False)
    opt = make_opt(module)
    A = spd(4, seed=5)
    original = th.linalg.eigh

    def failing_eigh(matrix, *args, **kwargs):
        if matrix.dtype == th.float32:
            raise RuntimeError("SSYEVD LWORK on illegal value")
        return original(matrix, *args, **kwargs)

    monkeypatch.setattr(M, "_SCIPY_EIGH_THRESHOLD", 10_000)
    monkeypatch.setattr(th.linalg, "eigh", failing_eigh)
    with pytest.warns(RuntimeWarning, match="Retrying with SciPy SYEVR"):
        vals, vecs, backend = opt._safe_symmetric_eigh(A, module=module, factor_name="A")
    assert backend == "scipy-evr"
    assert th.allclose((vecs * vals.unsqueeze(0)) @ vecs.t(), A, atol=1e-3)




def test_9_10_bounded_float64_fallback_is_the_last_resort(monkeypatch):
    import scipy.linalg as scipy_linalg

    module = nn.Linear(4, 2, bias=False)
    A = spd(4, seed=6)
    original = th.linalg.eigh

    def failing_eigh(matrix, *args, **kwargs):
        if matrix.dtype == th.float32:
            raise RuntimeError("SSYEVD LWORK")
        return original(matrix, *args, **kwargs)

    def failing_scipy(*args, **kwargs):
        raise ValueError("evr ei käytettävissä")

    monkeypatch.setattr(th.linalg, "eigh", failing_eigh)
    monkeypatch.setattr(scipy_linalg, "eigh", failing_scipy)

    opt = make_opt(module)
    monkeypatch.setattr(M, "_DENSE_EIGH_FALLBACK_MAX_SIZE", 8)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        vals, vecs, backend = opt._safe_symmetric_eigh(A, module=module, factor_name="A")
    assert backend == "torch-eigh-float64-cpu"
    assert th.allclose((vecs * vals.unsqueeze(0)) @ vecs.t(), A, atol=1e-3)

    monkeypatch.setattr(M, "_DENSE_EIGH_FALLBACK_MAX_SIZE", 2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(RuntimeError, match="fallback above n=2"):
            opt._safe_symmetric_eigh(A, module=module, factor_name="A")




def test_9_11_warnings_are_deduplicated():
    opt = make_opt(TinyNet())
    with pytest.warns(RuntimeWarning):
        opt._warn_once("sama viesti")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        opt._warn_once("sama viesti")
    assert caught == []




# ======================================================================
# 10. _update_inverses
# ======================================================================


def test_10_1_immature_statistics_block_the_refresh():
    module = nn.Linear(3, 2, bias=False)
    opt = make_opt(module, stats_accum_iter=5)
    feed_pairs(opt, module, [(th.randn(6, 3), th.randn(6, 2))])
    assert opt.module_states[module].stats_count == 1
    assert opt._update_inverses() == 0
    assert opt.module_states[module].A_eigvals is None


def test_10_2_mature_statistics_produce_a_full_spectral_generation():
    model = TinyNet(bias=True)
    opt = make_opt(model, stats_accum_iter=1)
    sample_through_hooks(opt, model, th.randn(8, 4))
    assert opt._update_inverses() == 2
    for module in opt.modules:
        state = opt.module_states[module]
        assert state.A_eigvecs is not None and state.G_eigvecs is not None
        assert state.A_eigvals.shape[0] == state.A.shape[0]
    assert "scipy-evr" in opt.last_eigh_backend


def test_10_3_a_failing_module_keeps_the_previous_generation(monkeypatch):
    model = TinyNet(bias=False)
    opt = make_opt(model, stats_accum_iter=1)
    sample_through_hooks(opt, model, th.randn(8, 4))
    assert opt._update_inverses() == 2
    old_vecs = opt.module_states[model.fc1].A_eigvecs.clone()

    original = opt._safe_symmetric_eigh

    def selective(matrix, *, module, factor_name):
        if module is model.fc2:
            raise RuntimeError("eigensolver kaatui")
        return original(matrix, module=module, factor_name=factor_name)

    monkeypatch.setattr(opt, "_safe_symmetric_eigh", selective)
    sample_through_hooks(opt, model, th.randn(8, 4))
    with pytest.warns(RuntimeWarning, match="Keeping the complete previous"):
        assert opt._update_inverses() == 0
    assert opt.eigh_failures == 1
    # Vanha sukupolvi säilyy kokonaisena: fc1 ei saanut uutta hajotelmaa.
    assert th.equal(opt.module_states[model.fc1].A_eigvecs, old_vecs)


def test_10_4_non_finite_factor_triggers_a_cold_restart():
    module = nn.Linear(3, 2, bias=False)
    opt = make_opt(module, stats_accum_iter=1)
    feed_pairs(opt, module, [(th.randn(6, 3), th.randn(6, 2))])
    opt.steps = 42
    opt.state[module.weight]["cold_momentum_buffer"] = th.zeros(2, 3)
    opt.module_states[module].A = th.full((3, 3), float("nan"))

    with pytest.raises(FloatingPointError, match="factor state was cleared"):
        opt._update_inverses()

    assert opt.eigh_failures == 1
    assert opt.steps == 0
    assert opt.stats_steps == 0
    assert opt.factor_steps == 0
    assert opt.module_states[module].A is None
    assert len(opt.state) == 0
    assert opt._steps_since_refresh == opt.kfac_update




def test_10_5_shape_corruption_is_caught_before_the_solver():
    module = nn.Linear(3, 2, bias=False)
    opt = make_opt(module, stats_accum_iter=1)
    feed_pairs(opt, module, [(th.randn(6, 3), th.randn(6, 2))])
    opt.module_states[module].A = th.eye(7)
    with pytest.warns(RuntimeWarning, match="Atomic K-FAC factor refresh failed"):
        assert opt._update_inverses() == 0


# ======================================================================
# 11. _register_refresh_failure -- toistuva epäonnistuminen on fataali
# ======================================================================


def test_11_1_failures_escalate_to_an_error(monkeypatch):
    monkeypatch.setattr(M, "_MAX_REFRESH_FAILURES", 2)
    opt = make_opt(TinyNet())
    with pytest.warns(RuntimeWarning, match="no new valid decompositions"):
        opt._register_refresh_failure()
    assert opt.consecutive_refresh_failures == 1
    with pytest.raises(RuntimeError, match="failed 2 times in a row"):
        opt._register_refresh_failure()




def test_11_3_a_successful_refresh_resets_the_counter():
    model = TinyNet(bias=False)
    opt = make_opt(model, stats_accum_iter=1, cold_iter=0)
    opt.consecutive_refresh_failures = 2
    sample_through_hooks(opt, model, th.randn(8, 4))
    for parameter in model.parameters():
        parameter.grad = th.randn_like(parameter)
    opt.step()  # kylmä askel
    for parameter in model.parameters():
        parameter.grad = th.randn_like(parameter)
    opt.step()  # ensimmäinen lämmin askel -> onnistunut refresh
    assert opt.factor_steps == 1
    assert opt.consecutive_refresh_failures == 0




# ======================================================================
# 12. Luonnollinen gradientti -- eksakti K-FAC-ratkaisu
# ======================================================================


def kfac_setup(damping: float = 0.05, **kwargs):
    module = nn.Linear(3, 2, bias=False).double()
    opt = make_opt(module, damping=damping, momentum=0.0, **kwargs)
    A = spd(3, seed=11, dtype=th.float64)
    G = spd(2, seed=12, dtype=th.float64)
    set_spectral_state(opt, module, A, G)
    return module, opt, A, G


def test_12_1_natural_gradient_solves_the_damped_kronecker_system():
    module, opt, A, G = kfac_setup(damping=0.05)
    grad = th.randn(2, 3, dtype=th.float64, generator=th.Generator().manual_seed(13))
    natural = opt._natural_gradient(module, grad)
    # Määritelmä: (A kron G + damping I) vec(X) = vec(grad), matriisimuodossa
    # G X A + damping X = grad.
    residual = G @ natural @ A + opt.damping * natural - grad
    assert residual.abs().max().item() < 1e-9


def test_12_2_natural_gradient_matches_an_explicit_kron_solve():
    module, opt, A, G = kfac_setup(damping=0.05)
    grad = th.randn(2, 3, dtype=th.float64, generator=th.Generator().manual_seed(14))
    natural = opt._natural_gradient(module, grad)
    system = th.kron(G, A) + opt.damping * th.eye(6, dtype=th.float64)
    solved = th.linalg.solve(system, grad.reshape(-1)).reshape(2, 3)
    assert th.allclose(natural, solved, atol=1e-9)


def test_12_3_damping_dominates_when_curvature_is_tiny():
    module = nn.Linear(3, 2, bias=False).double()
    opt = make_opt(module, damping=1.0, momentum=0.0)
    tiny = th.eye(3, dtype=th.float64) * 1e-12
    set_spectral_state(opt, module, tiny, th.eye(2, dtype=th.float64) * 1e-12)
    grad = th.randn(2, 3, dtype=th.float64, generator=th.Generator().manual_seed(15))
    natural = opt._natural_gradient(module, grad)
    assert th.allclose(natural, grad, atol=1e-6)


def test_12_4_missing_spectral_state_returns_none():
    module = nn.Linear(3, 2, bias=False)
    opt = make_opt(module)
    assert opt._natural_gradient(module, th.randn(2, 3)) is None
    state = opt.module_states[module]
    state.A_eigvals = th.ones(3)
    state.G_eigvals = th.ones(2)
    assert opt._natural_gradient(module, th.randn(2, 3)) is None  # vektorit puuttuvat


def test_12_5_non_positive_denominator_is_fatal():
    module, opt, _, _ = kfac_setup(damping=0.05)
    opt.module_states[module].G_eigvals = th.tensor([-1e6, 1.0], dtype=th.float64)
    with pytest.raises(FloatingPointError, match="Non-positive K-FAC denominator"):
        opt._natural_gradient(module, th.randn(2, 3, dtype=th.float64))


def test_12_6_non_finite_gradient_is_rejected():
    module, opt, _, _ = kfac_setup()
    grad = th.full((2, 3), float("nan"), dtype=th.float64)
    with pytest.raises(FloatingPointError, match="gradient matrix"):
        opt._natural_gradient(module, grad)


def test_12_8_grad_matrix_appends_the_bias_column():
    linear = nn.Linear(3, 2, bias=True)
    linear.weight.grad = th.ones(2, 3)
    linear.bias.grad = th.full((2,), 5.0)
    matrix = SB2KFACOptimizer._grad_matrix(linear)
    assert matrix.shape == (2, 4)
    assert th.all(matrix[:, -1] == 5.0)

    conv = nn.Conv2d(2, 3, kernel_size=2, bias=True)
    conv.weight.grad = th.ones_like(conv.weight)
    conv.bias.grad = th.full((3,), 7.0)
    conv_matrix = SB2KFACOptimizer._grad_matrix(conv)
    assert conv_matrix.shape == (3, 2 * 2 * 2 + 1)
    assert th.all(conv_matrix[:, -1] == 7.0)

    conv.bias.grad = None
    assert SB2KFACOptimizer._grad_matrix(conv) is None
    linear.weight.grad = None
    assert SB2KFACOptimizer._grad_matrix(linear) is None




# ======================================================================
# 13. step(): kylmä SGD, lämmin päivitys, KL-skaalaus, refresh-väli
# ======================================================================


def set_grads(model: nn.Module, value=None, seed: int = 0):
    gen = th.Generator().manual_seed(seed)
    for parameter in model.parameters():
        if value is None:
            parameter.grad = th.randn(
                parameter.shape, generator=gen, dtype=parameter.dtype
            )
        else:
            parameter.grad = th.full_like(parameter, value)


def test_13_1_cold_phase_length_is_cold_iter_plus_one():
    module = nn.Linear(3, 2, bias=False)
    opt = make_opt(module, cold_iter=2, momentum=0.9)
    observed = []
    for _ in range(5):
        observed.append(opt.next_step_is_cold())
        set_grads(module, value=1.0)
        opt.step()
        buffers = set(opt.state[module.weight])
        observed[-1] = (observed[-1], "cold_momentum_buffer" in buffers,
                        "warm_momentum_buffer" in buffers)
    # cold_iter=2 -> askeleet 1..3 ovat kylmiä (SB2: sgd_step > cold_iter).
    assert [flag for flag, _, _ in observed] == [True, True, True, False, False]
    assert [cold for _, cold, _ in observed] == [True] * 5
    assert [warm for _, _, warm in observed] == [False, False, False, True, True]
    assert opt.steps == 5


def test_13_2_cold_update_is_plain_momentum_sgd():
    module = nn.Linear(2, 2, bias=False)
    with th.no_grad():
        module.weight.zero_()
    lr, momentum = 0.1, 0.9
    opt = make_opt(module, lr=lr, momentum=momentum, cold_iter=10)
    grad = th.ones(2, 2)

    module.weight.grad = grad.clone()
    opt.step()
    assert th.allclose(module.weight, -lr * grad, atol=1e-7)

    module.weight.grad = grad.clone()
    opt.step()
    expected_velocity = momentum * grad + grad
    assert th.allclose(module.weight, -lr * grad - lr * expected_velocity, atol=1e-7)
    assert th.allclose(opt.state[module.weight]["cold_momentum_buffer"], expected_velocity)


def test_13_3_warm_step_without_factors_is_scaled_momentum_sgd():
    module = nn.Linear(2, 2, bias=False)
    with th.no_grad():
        module.weight.zero_()
    lr, momentum = 0.1, 0.9
    opt = make_opt(module, lr=lr, momentum=momentum, cold_iter=0, stats_accum_iter=1)
    grad = th.ones(2, 2)

    module.weight.grad = grad.clone()
    opt.step()  # kylmä
    cold_weight = module.weight.detach().clone()

    module.weight.grad = grad.clone()
    opt.step()  # lämmin, ei esikuntaajaa
    assert opt.preconditioner_is_active() is False
    # SB2:n lämmin haara käyttää lr*(1-momentum):ia ja omaa momentum-puskuria.
    assert th.allclose(module.weight, cold_weight - lr * (1 - momentum) * grad, atol=1e-7)


def _record_refresh_steps(opt, model, warm_steps: int):
    """Palauta ne optimizer-askeleet, joilla _update_inverses ajettiin."""
    calls = []
    original = opt._update_inverses

    def counting():
        result = original()
        calls.append(opt.steps + 1)
        return result

    opt._update_inverses = counting
    for _ in range(warm_steps):
        set_grads(model)
        opt.step()
    return calls


def test_13_4a_first_mature_warm_step_refreshes_immediately():
    model = TinyNet(bias=False)
    opt = make_opt(model, cold_iter=0, stats_accum_iter=1, kfac_update=3, momentum=0.0)
    sample_through_hooks(opt, model, th.randn(8, 4))
    calls = _record_refresh_steps(opt, model, warm_steps=9)

    # Askel 1 on kylmä. Ensimmäinen lämmin askel (2) päivittää HETI, koska
    # _steps_since_refresh alustetaan arvoon kfac_update -- ei odoteta, että
    # globaali laskuri osuu modulo-ehtoon.
    assert calls[0] == 2
    assert opt.factor_steps == len(calls)
    assert opt.preconditioner_is_active() is True
    assert all(b - a >= opt.kfac_update for a, b in zip(calls, calls[1:]))


def test_13_4b_refresh_spacing_equals_kfac_update():
    # Aiempi löydös (väli kfac_update+1) on korjattu v31:ssä.
    model = TinyNet(bias=False)
    opt = make_opt(model, cold_iter=0, stats_accum_iter=1, kfac_update=3, momentum=0.0)
    sample_through_hooks(opt, model, th.randn(8, 4))
    calls = _record_refresh_steps(opt, model, warm_steps=9)
    assert calls == [2, 5, 8]




def test_13_5_kl_clipping_and_natural_step_are_sb2_exact():
    module = nn.Linear(3, 2, bias=False).double()
    with th.no_grad():
        module.weight.zero_()
    lr, clip, eps = 0.1, 1e-3, 1e-8
    opt = make_opt(
        module, lr=lr, momentum=0.0, cold_iter=0, stats_accum_iter=1,
        kfac_clip=clip, eps=eps, damping=0.05,
    )
    set_spectral_state(opt, module, spd(3, seed=21, dtype=th.float64), spd(2, seed=22, dtype=th.float64))
    opt.stats_steps = 1

    grad = th.randn(2, 3, dtype=th.float64, generator=th.Generator().manual_seed(23))
    module.weight.grad = grad.clone()
    opt.step()  # kylmä askel
    weight_before = module.weight.detach().clone()

    module.weight.grad = grad.clone()
    opt.step()  # lämmin, esikuntaaja käytössä
    assert opt.factor_steps == 1

    natural = opt._natural_gradient(module, grad)
    vfv = float((grad * natural).sum().item()) * lr * lr
    expected_scale = min(1.0, math.sqrt(clip / (vfv + eps)))
    assert expected_scale < 1.0
    assert opt.last_v_f_v == pytest.approx(vfv, rel=1e-9)
    assert opt.last_kl_scale == pytest.approx(expected_scale, rel=1e-9)
    expected = weight_before - lr * (1.0 - opt.momentum) * expected_scale * natural
    assert th.allclose(module.weight, expected, atol=1e-9)


class NetWithFreeParameter(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3, 2, bias=False)
        self.log_std = nn.Parameter(th.zeros(2))

    def forward(self, x):
        return self.fc(x) * th.exp(self.log_std)


def test_13_7_unfactored_parameters_join_vfv_and_get_the_same_scaling():
    model = NetWithFreeParameter().double()
    with th.no_grad():
        model.log_std.zero_()
    lr = 0.1
    opt = make_opt(
        model, lr=lr, momentum=0.0, cold_iter=0, stats_accum_iter=1,
        kfac_clip=1e-3, damping=0.05,
    )
    set_spectral_state(opt, model.fc, spd(3, seed=41, dtype=th.float64), spd(2, seed=42, dtype=th.float64))
    opt.stats_steps = 1

    gen = th.Generator().manual_seed(43)
    weight_grad = th.randn(2, 3, dtype=th.float64, generator=gen)
    log_std_grad = th.randn(2, dtype=th.float64, generator=gen)

    for _ in range(2):
        model.fc.weight.grad = weight_grad.clone()
        model.log_std.grad = log_std_grad.clone()
        log_std_before = model.log_std.detach().clone()
        opt.step()

    natural = opt._natural_gradient(model.fc, weight_grad)
    expected_vfv = (
        float((weight_grad * natural).sum().item())
        + float((log_std_grad * log_std_grad).sum().item())
    ) * lr * lr
    assert opt.last_v_f_v == pytest.approx(expected_vfv, rel=1e-9)
    # Euklidinen fallback-parametri saa saman trust-region-skaalauksen ja
    # saman lr*(1-momentum)-kertoimen kuin faktoroidut kerrokset.
    expected_log_std = log_std_before - lr * (1.0 - opt.momentum) * opt.last_kl_scale * log_std_grad
    assert th.allclose(model.log_std, expected_log_std, atol=1e-12)


def test_13_9_negative_curvature_is_fatal(monkeypatch):
    module = nn.Linear(3, 2, bias=False).double()
    opt = make_opt(module, cold_iter=0, stats_accum_iter=1, damping=0.05)
    set_spectral_state(opt, module, spd(3, seed=61, dtype=th.float64), spd(2, seed=62, dtype=th.float64))
    opt.stats_steps = 1
    grad = th.randn(2, 3, dtype=th.float64, generator=th.Generator().manual_seed(63))

    module.weight.grad = grad.clone()
    opt.step()  # kylmä
    monkeypatch.setattr(opt, "_natural_gradient", lambda module, grad_matrix: -grad_matrix)
    module.weight.grad = grad.clone()
    with pytest.raises(FloatingPointError, match="curvature product must be non-negative"):
        opt.step()


def test_13_10_repeated_refresh_failures_stop_the_run(monkeypatch):
    monkeypatch.setattr(M, "_MAX_REFRESH_FAILURES", 2)
    module = nn.Linear(3, 2, bias=False)
    opt = make_opt(module, cold_iter=0, stats_accum_iter=1, momentum=0.0)
    feed_pairs(opt, module, [(th.randn(6, 3), th.randn(6, 2))])

    def failing(matrix, *, module, factor_name):
        raise RuntimeError("eigensolver kaatui")

    monkeypatch.setattr(opt, "_safe_symmetric_eigh", failing)
    set_grads(module, value=1.0)
    opt.step()  # kylmä
    weight_after_cold = module.weight.detach().clone()

    set_grads(module, value=1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        opt.step()  # lämmin, epäonnistuminen 1
    assert opt.consecutive_refresh_failures == 1

    set_grads(module, value=1.0)
    with pytest.raises(RuntimeError, match="failed 2 times in a row"):
        opt.step()
    assert opt.consecutive_refresh_failures == 2
    assert not th.equal(module.weight, weight_after_cold)  # 1. lämmin askel toteutui


def test_13_10b_immature_factors_are_not_a_refresh_failure():
    """Globaali portti auki mutta moduuli ei kypsä: ajoitustila, ei virhe."""
    module = nn.Linear(3, 2, bias=False)
    opt = make_opt(module, cold_iter=0, stats_accum_iter=5, momentum=0.0)
    opt.stats_steps = 5
    for _ in range(5):
        set_grads(module, value=1.0)
        opt.step()
    assert opt.consecutive_refresh_failures == 0
    assert opt.factor_steps == 0




def test_13_11_commit_rejects_duplicate_plans():
    module = nn.Linear(2, 2, bias=False)
    opt = make_opt(module)
    candidate = th.zeros(2, 2)
    with pytest.raises(RuntimeError, match="parameter was scheduled for update twice"):
        opt._commit_update_plans(
            [(module.weight, candidate), (module.weight, candidate)], []
        )
    with pytest.raises(RuntimeError, match="momentum buffer was scheduled twice"):
        opt._commit_update_plans(
            [], [(module.weight, "warm_momentum_buffer", candidate)] * 2
        )


def test_13_12_zero_momentum_keeps_no_buffers():
    module = nn.Linear(2, 2, bias=False)
    opt = make_opt(module, momentum=0.0, cold_iter=5, lr=0.1)
    with th.no_grad():
        module.weight.zero_()
    module.weight.grad = th.ones(2, 2)
    opt.step()
    assert th.allclose(module.weight, th.full((2, 2), -0.1), atol=1e-7)
    assert opt.state.get(module.weight, {}) == {}


def test_13_13_invalid_learning_rate_is_rejected():
    module = nn.Linear(2, 2, bias=False)
    opt = make_opt(module)
    module.weight.grad = th.ones(2, 2)
    opt.param_groups[0]["lr"] = float("nan")
    with pytest.raises(FloatingPointError, match="Invalid optimizer learning rate"):
        opt.step()


def test_13_14_modules_without_gradients_are_skipped():
    model = TinyNet(bias=False)
    opt = make_opt(model, cold_iter=0, momentum=0.0)
    fc1_before = model.fc1.weight.detach().clone()
    model.fc2.weight.grad = th.ones_like(model.fc2.weight)
    opt.step()  # kylmä
    model.fc2.weight.grad = th.ones_like(model.fc2.weight)
    opt.step()  # lämmin
    assert th.equal(model.fc1.weight, fc1_before)


def test_13_15_closure_is_evaluated_with_gradients_enabled():
    module = nn.Linear(2, 2, bias=False)
    opt = make_opt(module, cold_iter=5)
    seen = {}

    def closure():
        seen["grad_enabled"] = th.is_grad_enabled()
        module.weight.grad = th.ones(2, 2)
        return th.tensor(1.23)

    loss = opt.step(closure)
    assert seen["grad_enabled"] is True
    assert float(loss) == pytest.approx(1.23)


def test_13_16_bias_row_is_split_back_correctly():
    module = nn.Linear(3, 2, bias=True)
    with th.no_grad():
        module.weight.zero_()
        module.bias.zero_()
    opt = make_opt(module, lr=1.0, momentum=0.0, cold_iter=0, stats_accum_iter=1)
    module.weight.grad = th.ones(2, 3)
    module.bias.grad = th.full((2,), 4.0)
    opt.step()  # kylmä askel: molemmat päivittyvät erikseen
    assert th.allclose(module.weight, th.full((2, 3), -1.0))
    assert th.allclose(module.bias, th.full((2,), -4.0))


# ======================================================================
# 14. Checkpointointi
# ======================================================================


def trained_optimizer(stats_accum_iter: int = 1):
    model = TinyNet(bias=True)
    opt = make_opt(model, stats_accum_iter=stats_accum_iter)
    sample_through_hooks(opt, model, th.randn(8, 4))
    opt._update_inverses()
    opt.factor_steps = 1
    opt.steps = 17
    return model, opt




def fresh_target(**kwargs):
    model = TinyNet(bias=True)
    opt = make_opt(model, stats_accum_iter=kwargs.pop("stats_accum_iter", 1), **kwargs)
    return model, opt


def test_14_1_eigenvectors_are_not_saved():
    _, opt = trained_optimizer()
    sd = opt.state_dict()
    assert sd["kfac_audit_version"] == M.SB2_AUDIT_VERSION
    assert sd["kfac_steps"] == 17
    for entry in sd["kfac_module_stats"]:
        assert set(entry) == {"name", "weight_shape", "A", "G", "stats_count"}
        assert entry["A"] is not None




def test_14_2_covariance_only_checkpoint_rebuilds_the_basis():
    _, source = trained_optimizer()
    sd = copy.deepcopy(source.state_dict())

    model, target = fresh_target()
    target.load_state_dict(sd)

    assert target.steps == 17
    assert target.stats_steps == source.stats_steps
    for module in target.modules:
        state = target.module_states[module]
        assert state.A is not None and state.G is not None
        assert state.stats_count == 1
        assert state.A_eigvecs is None
    # Ilman ominaisvektoreita esikuntaajaa ei ole ennen ensimmäistä refreshiä.
    assert target.factor_steps == 0
    assert target.preconditioner_is_active() is False
    assert target._steps_since_refresh == target.kfac_update
    # ...mutta Fisher-historia säilyi, joten yksi refresh riittää.
    assert target._update_inverses() == 2
    assert target.module_states[model.fc1].A_eigvecs is not None




def test_14_4_load_does_not_mutate_the_callers_dict():
    _, source = trained_optimizer()
    sd = source.state_dict()
    keys_before = set(sd)
    _, target = fresh_target()
    target.load_state_dict(sd)
    assert set(sd) == keys_before
    assert "kfac_module_stats" in sd


def test_14_5_module_count_mismatch_resets_the_state():
    _, source = trained_optimizer()
    sd = copy.deepcopy(source.state_dict())
    sd["kfac_module_stats"] = sd["kfac_module_stats"][:1]

    model, target = fresh_target()
    with pytest.warns(UserWarning, match="do not match the current model"):
        target.load_state_dict(sd)
    assert target.module_states[model.fc1].A is None
    assert target.stats_steps == 0
    assert target.factor_steps == 0
    assert target.steps == 0


@pytest.mark.parametrize(
    "corrupt, message",
    [
        (lambda e: e.update(name="väärä-nimi"), "incompatible or invalid"),
        (lambda e: e.update(weight_shape=(9, 9)), "incompatible or invalid"),
        (lambda e: e.update(A=th.full_like(e["A"], float("nan"))), "incompatible or invalid"),
        (lambda e: e.update(G=None), "incompatible or invalid"),
        (lambda e: e.update(A=th.eye(9)), "incompatible or invalid"),
    ],
)
def test_14_6_corrupt_entries_are_rejected_atomically(corrupt, message):
    _, source = trained_optimizer()
    sd = copy.deepcopy(source.state_dict())
    corrupt(sd["kfac_module_stats"][1])

    model, target = fresh_target()
    with pytest.warns(UserWarning, match=message):
        target.load_state_dict(sd)
    # Myös ensimmäinen (ehjä) moduuli hylätään -- lataus on atominen.
    assert target.module_states[model.fc1].A is None
    assert target.stats_steps == 0


def test_14_7_counters_without_factors_are_reset():
    _, source = trained_optimizer()
    sd = copy.deepcopy(source.state_dict())
    del sd["kfac_module_stats"]

    _, target = fresh_target()
    with pytest.warns(UserWarning, match="missing or do not match"):
        target.load_state_dict(sd)
    assert target.stats_steps == 0
    assert target.factor_steps == 0




def test_14_8_stats_accum_iter_change_is_reported():
    _, source = trained_optimizer(stats_accum_iter=1)
    sd = copy.deepcopy(source.state_dict())
    _, target = fresh_target(stats_accum_iter=4)
    with pytest.warns(UserWarning, match="stats_accum_iter=1"):
        target.load_state_dict(sd)


def test_14_11_momentum_buffers_survive_the_round_trip():
    model, opt = trained_optimizer()
    set_grads(model, value=1.0)
    opt.momentum = 0.9
    opt.steps = 0
    opt.step()
    buffer = opt.state[model.fc1.weight]["cold_momentum_buffer"].clone()

    sd = copy.deepcopy(opt.state_dict())
    target_model, target = fresh_target()
    target.momentum = 0.9
    target.load_state_dict(sd)
    assert th.allclose(
        target.state[target_model.fc1.weight]["cold_momentum_buffer"], buffer
    )


# ======================================================================
# 15. ACKTR + SB3: rakennus, action spacet, runkoasettelu
# ======================================================================

import gymnasium as gym  # noqa: E402
from stable_baselines3.common.env_util import make_vec_env  # noqa: E402
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor  # noqa: E402


class DummyEnv(gym.Env):
    """Minimaalinen deterministinen ympäristö vaihtoehtoisille action spaceille."""

    def __init__(self, action_space: spaces.Space, image: bool = False):
        super().__init__()
        if image:
            self.observation_space = spaces.Box(0, 255, (3, 40, 40), dtype=np.uint8)
        else:
            self.observation_space = spaces.Box(-1.0, 1.0, (4,), dtype=np.float32)
        self.action_space = action_space
        self._image = image
        self._t = 0

    def _obs(self):
        if self._image:
            return np.full((3, 40, 40), self._t % 256, dtype=np.uint8)
        return np.full((4,), 0.01 * self._t, dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._t = 0
        return self._obs(), {}

    def step(self, action):
        self._t += 1
        return self._obs(), 1.0, False, self._t >= 10, {}


def make_dummy_env(action_space, image=False, n_envs=1):
    return make_vec_env(lambda: DummyEnv(action_space, image=image), n_envs=n_envs)


class RecurrentExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim: int = 8):
        super().__init__(observation_space, features_dim)
        n_input = int(np.prod(observation_space.shape))
        self.gru = nn.GRU(n_input, features_dim, batch_first=True)
        self.linear = nn.Linear(n_input, features_dim)

    def forward(self, observations):
        return self.linear(observations)


def build_model(**kwargs):
    params = dict(
        policy="MlpPolicy",
        env=make_vec_env("CartPole-v1", n_envs=2, seed=0),
        n_steps=8,
        seed=0,
        net_arch=(8,),
    )
    params.update(kwargs)
    return ACKTR(**params)


def test_15_1_default_construction_installs_the_kfac_optimizer():
    model = build_model()
    assert isinstance(model.policy.optimizer, SB2KFACOptimizer)
    # SB2 ACKTR:n oletus-lr on 0.25 ja se luetaan aikataulun alusta.
    assert model.policy.optimizer.param_groups[0]["lr"] == pytest.approx(0.25)
    assert model.gae_lambda == 1.0  # ei-GAE runner vastaa lambda=1:tä
    assert model.normalize_advantage is False




def test_15_2_defaults_are_the_sb2_constants():
    optimizer = build_model().policy.optimizer
    assert (optimizer.damping, optimizer.stats_decay, optimizer.momentum) == (0.01, 0.99, 0.9)
    assert (optimizer.kfac_clip, optimizer.kfac_update) == (0.001, 1)
    assert optimizer.cold_iter == 10
    # SB2 ACKTR ei aseta full_stats_init:iä, joten stats_accum_iter = cold_iter.
    assert optimizer.stats_accum_iter == 10




def test_15_3_user_constants_reach_the_optimizer():
    model = build_model(
        kfac_damping=0.5, cold_iter=3, kfac_momentum=0.5, kfac_stats_decay=0.5,
        kfac_clip=0.01, kfac_update=4,
    )
    optimizer = model.policy.optimizer
    assert optimizer.damping == 0.5
    assert optimizer.cold_iter == 3
    assert optimizer.stats_accum_iter == 3
    assert optimizer.momentum == 0.5
    assert optimizer.stats_decay == 0.5
    assert (optimizer.kfac_clip, optimizer.kfac_update) == (0.01, 4)




def test_15_6_action_space_support():
    multi = build_model(env=make_dummy_env(spaces.MultiDiscrete([2, 3])))
    assert isinstance(multi.action_space, spaces.MultiDiscrete)

    with pytest.raises(NotImplementedError, match="supports Discrete and Box"):
        build_model(env=make_dummy_env(spaces.MultiBinary(3)))




def test_15_10_reserved_policy_kwargs_are_stripped():
    with pytest.warns(UserWarning, match="Removed SB3-reserved entries"):
        model = build_model(policy_kwargs=dict(use_sde=True, lr_schedule=None))
    assert "use_sde" not in model.policy_kwargs
    assert "lr_schedule" not in model.policy_kwargs


def test_15_10b_trunk_entries_in_policy_kwargs_are_rejected():
    with pytest.raises(ValueError, match="not through policy_kwargs"):
        build_model(policy_kwargs=dict(net_arch=[8]))




def test_15_12_recurrent_modules_warn():
    with pytest.warns(UserWarning, match="recurrent modules"):
        build_model(features_extractor_class=RecurrentExtractor)




def test_15_14_prepare_actions_per_action_space():
    discrete = build_model()
    prepared = discrete._prepare_actions(th.tensor([[1.0], [0.0]]))
    assert prepared.dtype == th.long and prepared.shape == (2,)

    multi = build_model(env=make_dummy_env(spaces.MultiDiscrete([2, 3])))
    prepared_multi = multi._prepare_actions(th.tensor([[1.0, 2.0]]))
    assert prepared_multi.dtype == th.long and prepared_multi.shape == (1, 2)

    box = build_model(env=make_vec_env("Pendulum-v1", n_envs=1))
    actions = th.randn(3, 1)
    assert box._prepare_actions(actions) is actions




def test_15_15_rebuilding_the_model_detaches_old_hooks():
    model = build_model()
    first = model.policy.optimizer
    model._setup_model()
    assert first.handles == []
    assert model.policy.optimizer is not first
    assert len(model.policy.optimizer.handles) > 0


def test_15_16_learn_rejects_empty_budgets():
    model = build_model()
    with pytest.raises(ValueError, match="total_timesteps must be >= 1"):
        model.learn(total_timesteps=0)


def test_15_17_sb2_learning_rate_position():
    model = build_model(learning_rate=0.25, lr_schedule="linear")
    model._sb2_schedule_total_timesteps = 160
    model._sb2_lr_step = 0.0
    lr = model._set_sb2_learning_rate(16.0)
    # SB2 kutsuu Scheduler.value():a kerran per näyte ja käyttää VIIMEISTÄ
    # arvoa, eli aikataulupositiota start + len(obs) - 1.
    assert lr == pytest.approx(0.25 * (1.0 - 15.0 / 160.0))
    assert model.policy.optimizer.param_groups[0]["lr"] == pytest.approx(lr)
    lr2 = model._set_sb2_learning_rate(16.0)
    assert lr2 == pytest.approx(0.25 * (1.0 - 31.0 / 160.0))


# ======================================================================
# 16. ACKTR + SB3: oikea koulutus
# ======================================================================

NO_DUMP = 10 ** 9  # estää loggerin tyhjennyksen, jotta arvot voi lukea ajon jälkeen


def fast_kfac_model(**kwargs):
    """Nopeasti kypsyvä K-FAC-konfiguraatio testeille."""
    params = dict(
        cold_iter=1,
        kfac_update=1,
        learning_rate=0.25,
        lr_schedule="linear",
    )
    params.update(kwargs)
    return build_model(**params)




def test_16_1_short_cartpole_run_activates_the_preconditioner():
    model = fast_kfac_model()
    model.learn(total_timesteps=160, log_interval=NO_DUMP)
    optimizer = model.policy.optimizer
    values = model.logger.name_to_value

    assert model.num_timesteps >= 160
    assert optimizer.steps == 10          # 160 / (8 * 2) rollouttia, 1 askel kukin
    assert optimizer.stats_steps == 10    # SB2 kerää statistiikkaa myös kylmänä
    assert optimizer.factor_steps > 0
    assert optimizer.preconditioner_is_active() is True
    assert optimizer.eigh_failures == 0
    assert optimizer.consecutive_refresh_failures == 0
    assert values["train/kfac_preconditioner_active"] == 1
    assert values["train/optimizer_steps"] == 1
    assert values["train/kfac_kl_scale"] <= 1.0
    assert values["train/kfac_v_f_v"] >= 0.0
    assert math.isfinite(values["train/policy_loss"])
    assert math.isfinite(values["train/fisher_loss"])
    for parameter in model.policy.parameters():
        assert th.isfinite(parameter).all()




def test_16_4_max_grad_norm_is_clipped_only_in_the_cold_phase(monkeypatch):
    model = fast_kfac_model(max_grad_norm=0.5)
    optimizer = model.policy.optimizer
    original = th.nn.utils.clip_grad_norm_
    observed_steps = []

    def spy(parameters, max_norm, *args, **kwargs):
        observed_steps.append(optimizer.steps)
        assert max_norm == 0.5
        return original(parameters, max_norm, *args, **kwargs)

    monkeypatch.setattr(th.nn.utils, "clip_grad_norm_", spy)
    model.learn(total_timesteps=160, log_interval=NO_DUMP)

    assert optimizer.steps == 10
    # SB2 kutsuu clip_by_global_normia vain KfacOptimizerin kylmässä haarassa;
    # lämpimän haaran trust region hoituu vFv/kfac_clip-skaalauksella.
    assert observed_steps == list(range(optimizer.cold_iter + 1))




def test_16_6_continuous_action_space_run():
    model = fast_kfac_model(
        env=make_vec_env("Pendulum-v1", n_envs=2, seed=0), learning_rate=0.05
    )
    model.learn(total_timesteps=160, log_interval=NO_DUMP)
    values = model.logger.name_to_value
    assert model.policy.optimizer.factor_steps > 0
    assert math.isfinite(values["train/std"])
    # log_std ei kuulu Linear/Conv2d-lohkoon -> euklidinen fallback-polku.
    assert th.isfinite(model.policy.log_std).all()




def test_16_7_cnn_policy_exercises_the_conv_factor_path():
    model = fast_kfac_model(
        policy="CnnPolicy",
        env=make_dummy_env(spaces.Discrete(2), image=True, n_envs=1),
        n_steps=8,
        net_arch=(8,),
        shared_trunk=True,
        features_extractor_kwargs=dict(features_dim=32),
        learning_rate=0.05,
    )
    optimizer = model.policy.optimizer
    conv_modules = [m for m in optimizer.modules if isinstance(m, nn.Conv2d)]
    assert len(conv_modules) == 3  # yksi jaettu NatureCNN
    model.learn(total_timesteps=48, log_interval=NO_DUMP)
    assert optimizer.factor_steps > 0
    conv_state = optimizer.module_states[conv_modules[0]]
    expected_A, expected_G = optimizer._expected_factor_shapes(conv_modules[0])
    assert tuple(conv_state.A.shape) == expected_A
    assert tuple(conv_state.G.shape) == expected_G
    assert conv_state.A_eigvecs is not None
    assert th.isfinite(conv_modules[0].weight).all()




def test_16_8_learning_rate_follows_the_sb2_schedule():
    model = fast_kfac_model(learning_rate=0.25, lr_schedule="linear")
    model.learn(total_timesteps=160, log_interval=NO_DUMP)
    # 10 päivitystä x 16 näytettä = positio 159 / 160.
    assert model._sb2_lr_step == pytest.approx(160.0)
    assert model.logger.name_to_value["train/learning_rate"] == pytest.approx(
        0.25 * (1.0 - 159.0 / 160.0), rel=1e-6
    )


def test_16_9_continuing_a_run_does_not_rewind_the_schedule():
    model = fast_kfac_model(learning_rate=0.25, lr_schedule="linear")
    model.learn(total_timesteps=160, log_interval=NO_DUMP)
    lr_after_first = model.logger.name_to_value["train/learning_rate"]
    step_after_first = model._sb2_lr_step

    model.learn(total_timesteps=80, reset_num_timesteps=False, log_interval=NO_DUMP)
    assert model._sb2_schedule_total_timesteps == 240
    assert model._sb2_lr_step > step_after_first
    # Jatkoajo EI saa hypätä takaisin käyrän alkuun (0.25).
    assert model.logger.name_to_value["train/learning_rate"] < lr_after_first


def test_16_10_restarting_a_run_rewinds_the_schedule():
    model = fast_kfac_model(learning_rate=0.25, lr_schedule="linear")
    model.learn(total_timesteps=160, log_interval=NO_DUMP)
    model.learn(total_timesteps=160, reset_num_timesteps=True, log_interval=NO_DUMP)
    assert model._sb2_schedule_total_timesteps == 160
    assert model._sb2_lr_step == pytest.approx(160.0)


def test_16_12_training_is_deterministic_for_a_fixed_seed():
    def run():
        model = fast_kfac_model(env=make_vec_env("CartPole-v1", n_envs=2, seed=0), seed=7)
        model.learn(total_timesteps=96, log_interval=NO_DUMP)
        return [p.detach().clone() for p in model.policy.parameters()]

    first, second = run(), run()
    for a, b in zip(first, second):
        assert th.allclose(a, b, atol=1e-6, rtol=0)


def test_16_13_fisher_collection_is_released_when_training_fails(monkeypatch):
    model = fast_kfac_model()
    optimizer = model.policy.optimizer

    def exploding_step(*args, **kwargs):
        raise RuntimeError("simuloitu kaatuminen")

    monkeypatch.setattr(optimizer, "step", exploding_step)
    with pytest.raises(RuntimeError, match="simuloitu kaatuminen"):
        model.learn(total_timesteps=16, log_interval=NO_DUMP)
    assert optimizer.collecting_fisher_stats is False
    assert all(pairs == [] for pairs in optimizer._pending.values())


def test_16_14_sb3_save_load_round_trip(tmp_path):
    """SB3:n save/load vie K-FAC-tilan optimizerin state_dictin kautta."""
    model = fast_kfac_model()
    model.learn(total_timesteps=96, log_interval=NO_DUMP)
    source = model.policy.optimizer
    path = tmp_path / "acktr.zip"
    model.save(path)

    loaded = ACKTR.load(path, env=make_vec_env("CartPole-v1", n_envs=2, seed=0))
    optimizer = loaded.policy.optimizer
    assert isinstance(optimizer, SB2KFACOptimizer)
    assert optimizer.cold_iter == source.cold_iter
    assert optimizer.stats_accum_iter == source.stats_accum_iter

    for saved, restored in zip(model.policy.parameters(), loaded.policy.parameters()):
        assert th.allclose(saved, restored)
    for module_source, module_target in zip(source.modules, optimizer.modules):
        assert th.allclose(
            source.module_states[module_source].A,
            optimizer.module_states[module_target].A,
        )
    assert optimizer.steps == source.steps
    assert optimizer.stats_steps == source.stats_steps

    obs = model.env.reset()
    assert np.array_equal(
        model.predict(obs, deterministic=True)[0],
        loaded.predict(obs, deterministic=True)[0],
    )
    # Jatkokoulutus ladatulla mallilla rakentaa eigenbasiksen uudelleen.
    loaded.learn(total_timesteps=64, reset_num_timesteps=False, log_interval=NO_DUMP)
    assert optimizer.factor_steps > 0
    assert optimizer.eigh_failures == 0