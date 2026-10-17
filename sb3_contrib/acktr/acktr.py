# ACKTR (SB2-referenssi) -portti SB3:lle / PyTorchille — v31, jaettu verkko.
#
# YKSI VAI KAKSI VERKKOA: shared_trunk
# ------------------------------------
# shared_trunk=True (oletus) antaa politiikalle ja arvofunktiolle saman rungon;
# erillisiä ovat vain kaksi lineaarista päätä (action_net, value_net). Tämä on
# SB2 ACKTR:n oikea arkkitehtuuri sekä MlpPolicylle (jaetut 64-64-kerrokset)
# että CnnPolicylle (jaettu NatureCNN).
#
# shared_trunk=False rakentaa kaksi täysin erillistä runkoa samalla net_archilla,
# yhden politiikalle ja yhden arvofunktiolle. Tämä ei ole SB2:n arkkitehtuuri,
# mutta se on suoraviivaisempi optimoida: arvotappio ei enää vedä politiikan
# esitystä eikä K-FACin G-faktori sekoita kahden haaran gradientteja.
#
# Jaettu runko toteutetaan kahdella asetuksella:
#   * share_features_extractor -> SB3:n pi/vf-extractorit ovat sama olio
#     (True) tai kaksi erillistä instanssia (False).
#   * net_arch=[]              -> SB3:n MlpExtractor rakentaa AINA kaksi
#                                 erillistä haaraa (policy_net/value_net),
#                                 joten runko on siirrettävä features
#                                 extractoriin. Tyhjä net_arch tekee
#                                 molemmista haaroista Identityn.
# Runko elää siis SharedTrunkExtractorissa, ja _check_trunk_layout() tarkistaa
# ajonaikaisesti, että toteutunut asettelu vastaa pyydettyä.
#
# HUOM: pelkkä share_features_extractor EI ole runkoa ohjaava lippu. SB3:n
# oletus net_arch on dict(pi=[64,64], vf=[64,64]), joten MlpPolicylla runko on
# joka tapauksessa kahdennettu mlp_extractorissa ja features extractor on
# parametriton FlattenExtractor — lipun kääntäminen ei muuta yhtään painoa.
# Siksi valinta kulkee shared_trunkin kautta ja net_arch pakotetaan tyhjäksi.
#
# Runko on aina eksplisiittisesti määritelty kolmella ACKTR-argumentilla:
# features_extractor_class -> net_arch-mittainen MLP (activation_fn) -> päät.
# Mitään ei valita havaintoavaruuden perusteella net_arch-oletusta lukuun
# ottamatta: se on (64, 64), eli SB2:n MlpPolicy. Kuvahavainnoille SB2:n
# asettelu on net_arch=(), jolloin NatureCNN on koko runko.
#
# K-FAC:n kannalta tämä yksinkertaistaa kaiken. Jaetun rungon jokainen kerros
# on nyt tasan yhdellä forwardilla ja yhdellä Fisher-backwardilla, joten
# faktorit vastaavat SB2:n kaavoja suoraan (ks. "Fisher-semantiikka" alla).
#
# POISTETUT VALINNAT (36 parametria)
# ----------------------------------
# Arkkitehtuuri:  share_features_extractor, kfac_damping_scaling
# Moodit:         sb2_reference_mode, sb2_strict_kfac_defaults, use_kfac
# K-FAC-lisät:    use_stable_blend, kfac_alpha, kfac_max_natural_norm,
#                 clip_grad_norm_warm, kfac_stats_during_cold,
#                 kfac_full_stats_init, kfac_stats_accum_iter,
#                 optimizer_steps_per_rollout, kfac_max_refresh_failures,
#                 kfac_save_eigen_state
# Eigensolver:    kfac_eigen_backend, kfac_scipy_eigh_threshold,
#                 kfac_dense_eigh_fallback_max_size
# MPS:            kfac_mps_only, kfac_mps_eigh_on_cpu, kfac_mps_eigh_max_size,
#                 kfac_mps_allow_cpu_eigh_fallback
# SB2-yhteensopivuus: nprocs, n_cpu_tf_sess, async_eigen_decomp,
#                 full_tensorboard_log, apply_torch_thread_limit
# SB3-läpiviennit: use_rms_prop, rms_prop_eps, use_sde, sde_sample_freq,
#                 rollout_buffer_class, rollout_buffer_kwargs
# Duplikaatit:    kfac_colditer (= cold_iter), allow_multidiscrete
#
# Poistettujen valintojen tilalle jäi kussakin kohdassa yksi käyttäytyminen.
# Kun SB2 ja aiemman version ei-SB2-haara olivat eri mieltä, valinta on
# merkitty erikseen kommentilla kyseiseen kohtaan (ominaisarvojen nollaus,
# refresh-väli, lr-aikataulun kelaus).
#
# Fisher-semantiikka
# ------------------
# SB2 laskee tf.gradients(pg_fisher + vf_fisher, params) YHTENÄ backwardina,
# joten jaetun rungon kerroksen sisääntuleva gradientti on jo (g_pi + g_vf) ja
# G = B * sum (g_pi+g_vf)(g_pi+g_vf)^T ristitermeineen. Aiemmat versiot ajoivat
# kaksi erillistä backwardia ja summasivat lohkot vasta kovarianssivaiheessa,
# mikä pudotti ristitermit ja kaksinkertaisti rivimäärän. Tämä versio tekee
# yhden backwardin summatusta Fisher-lossista, eli vastaa SB2:ta täsmälleen ja
# on samalla yhden backward-passin nopeampi.
#
# Sama yksi backward palvelee molempia asetteluja ilman skaalauskorjauksia:
# jokainen faktoroitu kerros näkee tasan yhden (syote, gradientti) -parin ja
# G:n rivimäärä on B kummassakin tapauksessa. Ainoa ero on gradientin sisalto:
# jaetussa rungossa (g_pi + g_vf), erillisissa rungoissa g_pi tai g_vf.
# Juuri tästä syystä aiempien versioiden damping_scaling-koneisto on tarpeeton.
#
# SB2-lähteet: hill-a/stable-baselines master, acktr/acktr.py, acktr/kfac.py,
# acktr/kfac_utils.py, common/schedules.py, common/tf_util.py.
from __future__ import annotations

import functools
import math
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Type, Union

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces

from stable_baselines3 import A2C
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.preprocessing import is_image_space
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    NatureCNN,
)
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from stable_baselines3.common.utils import explained_variance

SB2_AUDIT_VERSION = "v31-shared-network"

# Suuret float32-matriisit CPU:lla reititetään SciPyn SYEVR-ajurille: PyTorchin
# CPU-eigh käyttää LAPACKin SSYEVD:tä, jonka workspace-kysely on rikki juuri
# NatureCNN:n kokoluokan (~3137x3137) matriiseilla.
_SCIPY_EIGH_THRESHOLD = 2048
# Viimeisen keinon float64-eigh on O(n^3) CPU:lla; ei ajeta mielivaltaisen
# suurille faktoreille.
_DENSE_EIGH_FALLBACK_MAX_SIZE = 1024
# Peräkkäisten epäonnistuneiden spektripäivitysten raja. Ilman tätä ajo jatkuisi
# hiljaisesti momentum-SGD:nä ACKTR:n learning ratella.
_MAX_REFRESH_FAILURES = 3


def mps_is_available() -> bool:
    backend = getattr(th.backends, "mps", None)
    if backend is None:
        return False
    try:
        return bool(backend.is_available())
    except Exception:
        return False


@functools.lru_cache(maxsize=1)
def mps_supports_reliable_eigh(tolerance: float = 2e-3) -> bool:
    """Onko torch.linalg.eigh MPS:llä sekä saatavilla että tarkka.

    Pelkkä saatavuus ei riitä: väärä ominaiskanta korruptoi jokaisen
    esikunnitetun päivityksen hiljaisesti sen sijaan että nostaisi virheen.
    Probe tarkistaa siksi myös rekonstruktiovirheen ja vertaa ominaisarvoja
    CPU:n LAPACKiin. Matriisi rakennetaan koskematta globaaliin RNG:hen, jotta
    probe ei häiritse siemenellisiä ajoja. Tulos välimuistitetaan, joten probe
    ajetaan korkeintaan kerran per prosessi.
    """
    if not mps_is_available():
        return False
    try:
        n = 12
        index = th.arange(n, dtype=th.float32)
        base = th.cos(index.unsqueeze(0) * index.unsqueeze(1) * 0.37)
        reference = base + base.t() + th.eye(n, dtype=th.float32) * float(2 * n)

        cpu_vals = th.linalg.eigvalsh(reference)
        vals, vecs = th.linalg.eigh(reference.to(device="mps"), UPLO="L")
        if not (th.isfinite(vals).all() and th.isfinite(vecs).all()):
            return False

        scale = float(reference.abs().max().item())
        rebuilt = (vecs * vals.unsqueeze(0)).matmul(vecs.t()).to(device="cpu")
        limit = tolerance * max(1.0, scale)
        return (
            float((rebuilt - reference).abs().max().item()) <= limit
            and float((vals.to(device="cpu") - cpu_vals).abs().max().item()) <= limit
        )
    except Exception:
        return False


def _default_base_extractor(
    observation_space: spaces.Space,
) -> Type[BaseFeaturesExtractor]:
    """SB3:n oma oletusvalinta havaintoavaruuden perusteella."""
    if isinstance(observation_space, spaces.Dict):
        return CombinedExtractor
    if is_image_space(observation_space):
        return NatureCNN
    return FlattenExtractor


class SharedTrunkExtractor(BaseFeaturesExtractor):
    """Actorin ja criticin JAETTU runko.

    SB3:n MlpExtractor rakentaa aina erilliset policy_net- ja value_net-haarat,
    joten jaettu runko täytyy toteuttaa features extractorina ja jättää
    net_arch tyhjäksi. Tämä luokka on siis se "yksi neuroverkko", jota sekä
    action_net että value_net käyttävät.

    Runko on kaksiosainen ja aina eksplisiittisesti määritelty:
        pohjaextractor  ->  trunk_arch-mittainen MLP  ->  action_net / value_net
    Pohjaextractor muuntaa havainnon vektoriksi (FlattenExtractor, NatureCNN,
    CombinedExtractor tai oma luokka); trunk_arch on sen päälle tuleva jaettu
    MLP. trunk_arch=() jättää MLP:n pois, jolloin pohjaextractor on koko runko —
    se on SB2:n CnnPolicyn asettelu, jossa NatureCNN päättyy jo 512-ulotteiseen
    lineaarikerrokseen.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        trunk_arch: Sequence[int],
        activation_fn: Type[nn.Module] = nn.Tanh,
        base_extractor_class: Optional[Type[BaseFeaturesExtractor]] = None,
        base_extractor_kwargs: Optional[Dict[str, Any]] = None,
    ):
        if trunk_arch is None:
            raise ValueError(
                "trunk_arch must be an explicit sequence of layer widths; use () "
                "for no MLP on top of the base extractor."
            )
        if base_extractor_class is None:
            base_extractor_class = _default_base_extractor(observation_space)
        base = base_extractor_class(
            observation_space, **(base_extractor_kwargs or {})
        )

        layers: list[nn.Module] = []
        last_dim = int(base.features_dim)
        for width in trunk_arch:
            width = int(width)
            if width < 1:
                raise ValueError(f"trunk_arch layer widths must be >= 1, got {width}.")
            layers.append(nn.Linear(last_dim, width))
            layers.append(activation_fn())
            last_dim = width

        super().__init__(observation_space, features_dim=last_dim)
        self.base = base
        self.trunk = nn.Sequential(*layers)

    def forward(self, observations) -> th.Tensor:
        return self.trunk(self.base(observations))


class ACKTRPolicy(ActorCriticPolicy):
    """ActorCriticPolicy, joka säilyttää runkoasettelun policy.save():ssa.

    SB3:n ActorCriticPolicy._get_constructor_parameters() EI sisällä
    share_features_extractoria, ja oletus on True. Erillisrunkoinen policy
    rakentuisi siis ActorCriticPolicy.load():ssa uudelleen jaettuna. Se ei nosta
    virhettä, koska jaetussa policyssa vf_features_extractor on sama olio kuin
    pi_features_extractor ja esiintyy state_dictissä omalla nimellään: avaimet
    täsmäävät, arvot eivät. Lopputulos on hiljainen korruptio, jossa criticin
    runko ylikirjoittaa actorin rungon. Mitattu: ladattu paino == vf, != pi.

    ACKTR.save()/load() ei kärsi tästä, koska se palauttaa policy_kwargsin
    checkpointista, mutta pelkkä policy-tason tallennus kärsii.

    Yksi luokka riittää kaikille kolmelle SB3-aliakselle, koska ne eroavat vain
    features_extractor_classin oletuksesta ja ACKTR asettaa sen aina itse.
    """

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(share_features_extractor=self.share_features_extractor)
        return data


@dataclass
class KFACModuleState:
    A: Optional[th.Tensor] = None
    G: Optional[th.Tensor] = None
    A_eigvals: Optional[th.Tensor] = None
    A_eigvecs: Optional[th.Tensor] = None
    G_eigvals: Optional[th.Tensor] = None
    G_eigvecs: Optional[th.Tensor] = None
    stats_count: int = 0


def make_sb2_lr_schedule(
    learning_rate: Union[float, Schedule],
    lr_schedule: str,
) -> Union[float, Schedule]:
    """SB2:n legacy Scheduler SB3:n progress_remaining-muodossa (1.0 -> 0.0)."""
    if callable(learning_rate):
        return learning_rate

    initial_lr = float(learning_rate)
    if initial_lr <= 0.0:
        raise ValueError("learning_rate must be positive.")
    name = lr_schedule.lower()
    if name not in {
        "constant",
        "linear",
        "double_linear_con",
        "middle_drop",
        "double_middle_drop",
    }:
        raise ValueError(
            f"Unknown lr_schedule={lr_schedule!r}. Supported: 'constant', "
            "'linear', 'double_linear_con', 'middle_drop', 'double_middle_drop'."
        )

    def schedule(progress_remaining: float) -> float:
        remaining = float(progress_remaining)
        if name == "constant":
            return initial_lr
        if name == "linear":
            return initial_lr * remaining
        # SB2:n Scheduler saa kuluneen osuuden, SB3 antaa jäljellä olevan.
        elapsed = 1.0 - remaining
        if name == "double_linear_con":
            return initial_lr * max(0.125, 1.0 - 2.0 * elapsed)
        if name == "middle_drop":
            return initial_lr * (0.075 if remaining < 0.75 else remaining)
        # double_middle_drop
        if remaining < 0.25:
            return initial_lr * 0.125
        if remaining < 0.75:
            return initial_lr * 0.075
        return initial_lr * remaining

    return schedule


class SB2KFACOptimizer(th.optim.Optimizer):
    """SB2 ACKTR:n K-FAC-optimizer PyTorchille.

    SB2-vastaavuudet:
    - cold_iter: ensimmäiset askeleet tavallisena momentum-SGD:nä
    - stats_accum_iter: monenko tasapainoisen näytteen keskiarvo alustaa faktorin
    - kfac_clip: trust region -skaalaus min(1, sqrt(kfac_clip / vFv))
    - kfac_update: spektrifaktorien päivitysväli
    - momentum: TF:n MomentumOptimizer(lr*(1-momentum), momentum) lämpimässä
      haarassa, MomentumOptimizer(lr, momentum) kylmässä
    - damping vastaa SB2:n KFAC epsilon/damping-ideaa

    Rajoitukset (samat kuin aiemmissa versioissa):
    - ei TF-graafin operaatiotason MatMul/Conv2D/BiasAdd-porttia
    - ei async_eigen_decompia
    - ei recurrent-policyjen erikoiskäsittelyä; vain Linear ja Conv2d
      faktoroidaan, muut parametrit menevät euklidista momentum-polkua

    Apple MPS: faktorien keruu ja parametripäivitys pysyvät MPS:llä.
    Ominaisarvohajotelma ajetaan MPS:llä jos ajonaikainen probe toteaa sen
    tarkaksi, muuten CPU:n kautta ja tulokset kopioidaan takaisin.
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 0.25,
        damping: float = 0.01,
        stats_decay: float = 0.99,
        kfac_update: int = 1,
        kfac_clip: float = 0.001,
        momentum: float = 0.9,
        cold_iter: int = 10,
        # SB2 KfacOptimizer.__init__ päättyy riveihin
        #     if not self._full_stats_init:
        #         self._stats_accum_iter = self._cold_iter
        # eikä ACKTR koskaan välitä full_stats_init:iä, joten SB2 ACKTR:n
        # todellinen arvo on cold_iter (10), ei konstruktorin oletus 60.
        stats_accum_iter: int = 10,
        eps: float = 1e-8,
        debug_finite_checks: bool = False,
    ):
        params = [p for p in model.parameters() if p.requires_grad]
        if not params:
            raise ValueError("Model has no trainable parameters.")
        super().__init__(params, defaults={"lr": lr})

        self.model = model
        self.damping = float(damping)
        self.stats_decay = float(stats_decay)
        self.kfac_update = max(1, int(kfac_update))
        self.kfac_clip = float(kfac_clip)
        self.momentum = float(momentum)
        self.cold_iter = max(0, int(cold_iter))
        self.stats_accum_iter = max(1, int(stats_accum_iter))
        self.eps = float(eps)
        self.debug_finite_checks = bool(debug_finite_checks)

        if not 0.0 <= self.stats_decay < 1.0:
            raise ValueError("stats_decay must be in [0, 1).")
        if not 0.0 <= self.momentum < 1.0:
            raise ValueError("momentum must be in [0, 1).")
        if self.damping <= 0.0:
            raise ValueError("damping must be positive.")
        if self.kfac_clip <= 0.0:
            raise ValueError("kfac_clip must be positive.")
        if self.eps <= 0.0:
            raise ValueError("eps must be positive.")

        self.steps = 0
        self.factor_steps = 0
        self.stats_steps = 0
        self.collecting_fisher_stats = False
        self.last_v_f_v = 0.0
        self.last_kl_scale = 1.0
        self.eigh_failures = 0
        self.consecutive_refresh_failures = 0
        self.last_eigh_backend = "none"
        self.mps_eigh_native: Optional[bool] = None
        self._warned_messages: set[str] = set()
        # Alustetaan kfac_updateen, jotta ensimmäinen kypsä lämmin askel
        # päivittää spektrin heti eikä jää odottamaan globaalin laskurin
        # osumista modulon kanssa.
        self._steps_since_refresh = self.kfac_update

        self.modules: list[nn.Module] = []
        self.module_states: dict[nn.Module, KFACModuleState] = {}
        self._pending: dict[nn.Module, list[tuple[th.Tensor, th.Tensor]]] = {}
        self.handles: list[Any] = []
        # Nimet esilasketaan: named_modules()-skannaus per kutsu teki
        # lokituksesta O(moduulit^2).
        self._module_names: dict[nn.Module, str] = {}

        seen_params: dict[int, str] = {}
        for module_name, module in model.named_modules():
            if not isinstance(module, (nn.Linear, nn.Conv2d)):
                continue
            # Kronecker-lohko ankkuroidaan painomatriisiin; pelkän biaksen
            # sisältävät moduulit hoituvat euklidisella varapolulla.
            if not module.weight.requires_grad:
                continue
            if isinstance(module, nn.Conv2d):
                if module.groups != 1:
                    raise NotImplementedError(
                        "Grouped/depthwise Conv2d is not supported by this K-FAC "
                        f"port; got groups={module.groups} for {module}."
                    )
                if isinstance(module.padding, str):
                    raise NotImplementedError(
                        "String Conv2d padding is not supported by the K-FAC "
                        f"factor extractor; got padding={module.padding!r}."
                    )
                if module.padding_mode != "zeros":
                    raise NotImplementedError(
                        "K-FAC Conv2d factors support only zero padding; got "
                        f"padding_mode={module.padding_mode!r}."
                    )

            for param_name, param in module.named_parameters(recurse=False):
                if not param.requires_grad:
                    continue
                current = f"{module_name}.{param_name}" if module_name else param_name
                previous = seen_params.get(id(param))
                if previous is not None:
                    raise NotImplementedError(
                        "The same trainable parameter is owned by two Linear/Conv2d "
                        f"modules ({previous!r} and {current!r}); SB2-style factor "
                        "assignment is ambiguous for tied weights."
                    )
                seen_params[id(param)] = current

            self.modules.append(module)
            self.module_states[module] = KFACModuleState()
            self._pending[module] = []
            self._module_names[module] = module_name or "<root>"
            self.handles.append(module.register_forward_hook(self._capture_forward))

    # ------------------------------------------------------------------
    # Elinkaari
    # ------------------------------------------------------------------

    def remove_hooks(self) -> None:
        """Irrota kaikki forward-hookit.

        Hookit muodostavat viittaussyklin moduuli -> optimizer -> malli ->
        moduuli, joten __del__:iin luottaminen jättää ne elämään täyteen
        GC-kierrokseen asti.
        """
        for handle in self.handles:
            try:
                handle.remove()
            except Exception:
                pass
        self.handles = []
        self.collecting_fisher_stats = False
        for pairs in self._pending.values():
            pairs.clear()

    def __del__(self):
        try:
            self.remove_hooks()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Apurit
    # ------------------------------------------------------------------

    def module_name(self, module: nn.Module) -> str:
        return self._module_names.get(module, repr(module))

    def modules_without_statistics(self) -> list[str]:
        """Faktoroidut moduulit, jotka eivät ole saaneet yhtään Fisher-näytettä.

        Tällainen moduuli ei koskaan tule esikunnitetuksi vaan putoaa hiljaisesti
        euklidiselle momentum-polulle, eikä siitä jää muuta jälkeä lokeihin.
        """
        return [
            self.module_name(module)
            for module in self.modules
            if int(self.module_states[module].stats_count) <= 0
        ]

    @staticmethod
    def _uses_trainable_bias(module: nn.Module) -> bool:
        bias = getattr(module, "bias", None)
        return bias is not None and bool(bias.requires_grad)

    def _expected_factor_shapes(
        self, module: nn.Module
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        if isinstance(module, nn.Linear):
            in_dim = int(module.in_features)
            out_dim = int(module.out_features)
        elif isinstance(module, nn.Conv2d):
            kernel_h, kernel_w = self._conv_pair(module.kernel_size)
            in_dim = int(module.in_channels * kernel_h * kernel_w)
            out_dim = int(module.out_channels)
        else:
            raise TypeError(f"Unsupported K-FAC module: {type(module).__name__}")
        if self._uses_trainable_bias(module):
            in_dim += 1
        return (in_dim, in_dim), (out_dim, out_dim)

    @staticmethod
    def _require_finite(tensor: th.Tensor, description: str) -> None:
        if not th.isfinite(tensor).all():
            raise FloatingPointError(f"Non-finite values in {description}.")

    def _check_large_finite(self, tensor: th.Tensor, description: str) -> None:
        """Äärellisyystarkistus kovarianssin kokoiselle tensorille.

        Yksi host-device-synkronointi miljoonien alkioiden yli. Epääärellinen
        kovarianssi jää joka tapauksessa kiinni ennen päivitystä, koska
        _safe_symmetric_eigh validoi hajotettavan matriisin ja _natural_gradient
        oman tuloksensa.
        """
        if self.debug_finite_checks:
            self._require_finite(tensor, description)

    def _warn_once(self, message: str) -> None:
        if message not in self._warned_messages:
            warnings.warn(message, RuntimeWarning)
            self._warned_messages.add(message)

    def _restart_kfac_state(self, *, clear_optimizer_state: bool) -> None:
        for module in self.modules:
            self.module_states[module] = KFACModuleState()
            self._pending[module].clear()
        self.steps = 0
        self.factor_steps = 0
        self.stats_steps = 0
        self._steps_since_refresh = self.kfac_update
        self.collecting_fisher_stats = False
        self.last_v_f_v = 0.0
        self.last_kl_scale = 1.0
        self.last_eigh_backend = "none"
        self.consecutive_refresh_failures = 0
        if clear_optimizer_state:
            self.state.clear()

    # ------------------------------------------------------------------
    # Fisher-statistiikan keruu
    # ------------------------------------------------------------------

    def begin_fisher_stats_collection(self) -> None:
        """Aloita yksi Fisher-näyte.

        Kutsujan on ajettava tasan YKSI backward summatusta Fisher-lossista
        (pg_fisher + vf_fisher), kuten SB2 tekee. Jokainen faktoroitu moduuli
        saa tällöin tarkalleen yhden (input, grad_output) -parin.
        """
        if self.collecting_fisher_stats:
            raise RuntimeError("Fisher statistics collection is already active.")
        self.collecting_fisher_stats = True
        for pairs in self._pending.values():
            pairs.clear()

    def discard_fisher_stats_collection(self) -> None:
        self.collecting_fisher_stats = False
        for pairs in self._pending.values():
            pairs.clear()

    def _capture_forward(
        self,
        module: nn.Module,
        inputs: tuple[th.Tensor, ...],
        output: th.Tensor,
    ) -> None:
        """Talleta forward-syöte ja kiinnitä hook sen ulostulon gradienttiin.

        Tensorihook välttää PyTorchin full-backward-hook-varoituksen silloin kun
        moduulin syötteet (havainnot/aktivaatiot) eivät vaadi gradienttia.
        Talletettu gradientti on täsmälleen moduulin ulostulon backprop-faktori.
        """
        if not self.collecting_fisher_stats:
            return
        if not inputs or not isinstance(inputs[0], th.Tensor):
            return
        if not isinstance(output, th.Tensor) or not output.requires_grad:
            return

        x = inputs[0].detach()
        pending = self._pending[module]

        def capture_grad(grad: th.Tensor) -> th.Tensor:
            if self.collecting_fisher_stats:
                pending.append((x, grad.detach()))
            return grad

        output.register_hook(capture_grad)

    @staticmethod
    def _conv_pair(value: Union[int, tuple[int, int]]) -> tuple[int, int]:
        if isinstance(value, tuple):
            if len(value) != 2:
                raise ValueError(f"Expected a pair, got {value!r}.")
            return int(value[0]), int(value[1])
        return int(value), int(value)

    @th.no_grad()
    def _linear_factors(
        self, module: nn.Linear, x: th.Tensor, g: th.Tensor
    ) -> tuple[th.Tensor, th.Tensor]:
        x2 = x.reshape(-1, x.shape[-1])
        g2 = g.reshape(-1, g.shape[-1])
        if x2.shape[0] != g2.shape[0]:
            raise RuntimeError(
                f"Linear K-FAC row mismatch for {module}: inputs={tuple(x2.shape)}, "
                f"grad_outputs={tuple(g2.shape)}."
            )
        if self._uses_trainable_bias(module):
            ones = th.ones(x2.shape[0], 1, device=x2.device, dtype=x2.dtype)
            x2 = th.cat([x2, ones], dim=1)
        rows = float(x2.shape[0])
        if rows <= 0:
            raise RuntimeError(f"Empty Linear Fisher sample for {module}.")
        # SB2 KFAC saa gradientit keskiarvoistetuista Fisher-losseista ja kertoo
        # backward-faktorin näytemäärällä ennen kovarianssia; skaalaukset
        # supistuvat muotoon rows * sum(g g^T).
        return x2.t().matmul(x2) / rows, g2.t().matmul(g2) * rows

    @th.no_grad()
    def _conv2d_factors(
        self, module: nn.Conv2d, x: th.Tensor, g: th.Tensor
    ) -> tuple[th.Tensor, th.Tensor]:
        if x.dim() != 4 or g.dim() != 4:
            raise RuntimeError(
                f"Conv2d K-FAC expects NCHW tensors, got x={tuple(x.shape)}, "
                f"g={tuple(g.shape)} for {module}."
            )
        grad_spatial = int(g.shape[2] * g.shape[3])
        if grad_spatial <= 0:
            raise RuntimeError(f"Invalid Conv2d output shape {tuple(g.shape)}.")

        g_rows = g.permute(0, 2, 3, 1).reshape(-1, g.shape[1])
        patches = F.unfold(
            x,
            kernel_size=module.kernel_size,
            dilation=module.dilation,
            padding=module.padding,
            stride=module.stride,
        )
        spatial = int(patches.shape[-1])
        if spatial != grad_spatial:
            raise RuntimeError(
                f"Conv2d K-FAC spatial mismatch for {module}: unfold={spatial}, "
                f"grad_output={grad_spatial}."
            )

        rows = patches.transpose(1, 2).reshape(-1, patches.shape[1]) / float(spatial)
        if self._uses_trainable_bias(module):
            bias = th.ones(
                rows.shape[0], 1, device=rows.device, dtype=rows.dtype
            ) / float(spatial)
            rows = th.cat([rows, bias], dim=1)

        batch = int(x.shape[0])
        if batch <= 0:
            raise RuntimeError(f"Empty Conv2d Fisher sample for {module}.")
        # SB2:n T1 skaalaa ulostulon backpropit B*T:llä ja jakaa kovarianssin
        # B*T:llä, jolloin jäljelle jää (B*T) * sum(g g^T).
        A = rows.t().matmul(rows) / float(batch)
        G = g_rows.t().matmul(g_rows) * float(batch * spatial)
        del patches, rows
        return A, G

    @th.no_grad()
    def complete_fisher_stats_collection(self) -> None:
        """Kirjaa yksi SB2-tyylinen statistiikka-askel atomisesti.

        SB2 alustaa uuden faktorin stats_accum_iter:n tasapainoisesta näytteestä
        (kukin painolla 1/stats_accum_iter) ja siirtyy vasta sen jälkeen
        eksponentiaaliseen liukuvaan keskiarvoon. Laskuri pidetään
        moduulikohtaisena, jotta uudet tai checkpointista palautetut faktorit
        käyvät saman alustuksen läpi.
        """
        if not self.collecting_fisher_stats:
            raise RuntimeError("Fisher statistics collection is not active.")

        try:
            staged: list[tuple[nn.Module, th.Tensor, th.Tensor]] = []
            for module in self.modules:
                pairs = self._pending[module]
                if not pairs:
                    continue
                if len(pairs) > 1:
                    raise NotImplementedError(
                        f"{self.module_name(module)} received {len(pairs)} Fisher "
                        "backprop tensors in one statistics step. This port expects "
                        "one summed Fisher backward through each layer; reused or "
                        "recurrent layers are not supported."
                    )
                x, g = pairs[0]
                if isinstance(module, nn.Linear):
                    A, G = self._linear_factors(module, x, g)
                elif isinstance(module, nn.Conv2d):
                    A, G = self._conv2d_factors(module, x, g)
                else:
                    continue
                # Vapauta aktivaatiot heti kun lohkot ovat olemassa; unfold-
                # puskurit ovat muistin huippukohta.
                pairs.clear()

                expected_A, expected_G = self._expected_factor_shapes(module)
                if tuple(A.shape) != expected_A or tuple(G.shape) != expected_G:
                    raise RuntimeError(
                        f"Unexpected K-FAC factor shape for {module}: "
                        f"A={tuple(A.shape)} expected={expected_A}, "
                        f"G={tuple(G.shape)} expected={expected_G}."
                    )
                self._check_large_finite(A, f"K-FAC A factor for {module}")
                self._check_large_finite(G, f"K-FAC G factor for {module}")
                staged.append((module, A, G))

            if not staged:
                raise RuntimeError(
                    "No Linear/Conv2d Fisher factors were collected; the policy may "
                    "use unsupported trainable operations."
                )

            # Rakenna korvaavat tensorit valmiiksi, jotta commit pysyy atomisena
            # vaikka allokointi tai aritmetiikka kaatuisi myöhemmällä kerroksella.
            replacements: list[tuple[nn.Module, th.Tensor, th.Tensor, int]] = []
            for module, A, G in staged:
                state = self.module_states[module]
                if state.A is not None and state.A.shape != A.shape:
                    raise RuntimeError(f"K-FAC A shape changed for {module}.")
                if state.G is not None and state.G.shape != G.shape:
                    raise RuntimeError(f"K-FAC G shape changed for {module}.")

                count = max(0, int(state.stats_count))
                if count < self.stats_accum_iter:
                    prior_A = th.zeros_like(A) if state.A is None else state.A
                    prior_G = th.zeros_like(G) if state.G is None else state.G
                    new_A = prior_A + A / float(self.stats_accum_iter)
                    new_G = prior_G + G / float(self.stats_accum_iter)
                else:
                    if state.A is None or state.G is None:
                        raise RuntimeError(f"Invalid mature K-FAC state for {module}.")
                    new_A = state.A * self.stats_decay + A * (1.0 - self.stats_decay)
                    new_G = state.G * self.stats_decay + G * (1.0 - self.stats_decay)

                self._check_large_finite(new_A, f"updated A factor for {module}")
                self._check_large_finite(new_G, f"updated G factor for {module}")
                replacements.append((module, new_A, new_G, count + 1))

            for module, new_A, new_G, new_count in replacements:
                state = self.module_states[module]
                state.A = new_A
                state.G = new_G
                state.stats_count = new_count
            self.stats_steps += 1
        finally:
            self.discard_fisher_stats_collection()

    # ------------------------------------------------------------------
    # Spektrifaktorit
    # ------------------------------------------------------------------

    def _use_mps_eigh(self) -> bool:
        if self.mps_eigh_native is None:
            self.mps_eigh_native = mps_supports_reliable_eigh()
            if not self.mps_eigh_native:
                self._warn_once(
                    "torch.linalg.eigh is unavailable or numerically unreliable on "
                    "this MPS build, so K-FAC factors are decomposed on the CPU and "
                    "copied back. Factor accumulation and parameter updates stay "
                    "on MPS."
                )
        return self.mps_eigh_native

    @th.no_grad()
    def _safe_symmetric_eigh(
        self, matrix: th.Tensor, *, module: nn.Module, factor_name: str
    ) -> tuple[th.Tensor, th.Tensor, str]:
        """Robusti ominaisarvohajotelma symmetriselle K-FAC-faktorille.

        Suuria float32-matriiseja ei anneta CPU:lla torch.linalg.eigh:lle:
        PyTorchin CPU-polku käyttää LAPACKin SSYEVD:tä, jonka LWORK-kysely on
        useissa julkaisuissa raportoitu rikkinäiseksi juuri NatureCNN:n
        tuottamalla kokoluokalla. SciPyn eigh(driver='evr') käyttää SYEVR-perhettä.
        """
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError(
                f"{factor_name} factor for {module} must be square, got "
                f"{tuple(matrix.shape)}."
            )
        n = int(matrix.shape[0])
        if n < 1:
            raise ValueError(f"{factor_name} factor for {module} is empty.")

        symmetric = (0.5 * (matrix + matrix.transpose(-2, -1))).detach().contiguous()
        # Ehdoton tarkistus: ajetaan kerran per refresh-väli, ei per askel, ja on
        # viimeinen mahdollisuus napata myrkytetty faktori ennen kuin
        # eigensolver muuttaa sen hiljaiseksi roskaksi.
        if not th.isfinite(symmetric).all():
            finite = symmetric[th.isfinite(symmetric)]
            finite_min = float(finite.min().item()) if finite.numel() else float("nan")
            finite_max = float(finite.max().item()) if finite.numel() else float("nan")
            raise FloatingPointError(
                f"Non-finite values in K-FAC {factor_name} factor for {module}; "
                f"shape={tuple(symmetric.shape)}, finite_min={finite_min}, "
                f"finite_max={finite_max}."
            )

        source_device = symmetric.device
        force_cpu = source_device.type == "mps" and not self._use_mps_eigh()
        work = symmetric.to(device="cpu").contiguous() if force_cpu else symmetric
        suffix = "-for-mps" if source_device.type == "mps" else ""

        use_scipy = work.device.type == "cpu" and n >= _SCIPY_EIGH_THRESHOLD
        if not use_scipy:
            try:
                vals, vecs = th.linalg.eigh(work, UPLO="L")
                return (
                    vals.to(device=matrix.device, dtype=matrix.dtype),
                    vecs.to(device=matrix.device, dtype=matrix.dtype),
                    f"torch-{work.device.type}{suffix}",
                )
            except RuntimeError as exc:
                if work.device.type == "mps":
                    # Probe meni läpi mutta tämä koko kaatui. Älä yritä
                    # laitepolkua uudelleen jokaisella seuraavalla faktorilla.
                    self.mps_eigh_native = False
                    work = symmetric.to(device="cpu").contiguous()
                self._warn_once(
                    f"torch.linalg.eigh failed for K-FAC {factor_name} factor "
                    f"{tuple(symmetric.shape)} ({type(exc).__name__}: {exc}). "
                    "Retrying with SciPy SYEVR on CPU."
                )
                use_scipy = True

        if use_scipy:
            try:
                import numpy as np
                from scipy import linalg as scipy_linalg

                # Pidä float32: SYEVR välttää rikkinäisen SSYEVD-ajurin ja vie
                # noin puolet float64-kopion muistista.
                cpu_matrix = work.to(device="cpu").contiguous()
                array = np.array(cpu_matrix.numpy(), order="F", copy=True)
                vals_np, vecs_np = scipy_linalg.eigh(
                    array, lower=True, check_finite=False, overwrite_a=True, driver="evr"
                )
                return (
                    th.from_numpy(np.ascontiguousarray(vals_np)).to(
                        device=matrix.device, dtype=matrix.dtype
                    ),
                    th.from_numpy(np.ascontiguousarray(vecs_np)).to(
                        device=matrix.device, dtype=matrix.dtype
                    ),
                    f"scipy-evr{suffix}",
                )
            except Exception as scipy_exc:
                self._warn_once(
                    f"SciPy SYEVR failed for K-FAC {factor_name} factor "
                    f"{tuple(symmetric.shape)} ({type(scipy_exc).__name__}: "
                    f"{scipy_exc}). Retrying with a bounded float64 CPU eigh."
                )

        # Viimeinen keino. Yleinen SVD ei takaa etumerkillistä hajotelmaa lievästi
        # indefiniitille symmetriselle faktorille, joten käytetään rajattua
        # float64-eigh:ta CPU:lla eikä hyväksytä rajatonta kuutiollista ratkaisua.
        if n > _DENSE_EIGH_FALLBACK_MAX_SIZE:
            raise RuntimeError(
                f"All K-FAC eigensolvers failed for {factor_name} factor "
                f"{tuple(symmetric.shape)} of {module}; refusing the float64 CPU "
                f"fallback above n={_DENSE_EIGH_FALLBACK_MAX_SIZE}."
            )
        vals64, vecs64 = th.linalg.eigh(
            symmetric.to(device="cpu", dtype=th.float64).contiguous(), UPLO="L"
        )
        return (
            vals64.to(device=matrix.device, dtype=matrix.dtype),
            vecs64.to(device=matrix.device, dtype=matrix.dtype),
            f"torch-eigh-float64-cpu{suffix}",
        )

    def _clamp_eigenvalues(self, vals: th.Tensor) -> th.Tensor:
        """Nollaa numeerisesti merkityksettömät ominaisarvot.

        VALINTA: SB2:n kfac_utils.detect_min_val käyttää kiinteää absoluuttista
        rajaa 1e-6. Se on mielivaltainen faktorille, jonka skaalaa ei tunneta:
        pienimagnitudisella kovarianssilla se leikkaa lähes kaiken ja suurella
        ei mitään. Tässä käytetään standardia suhteellista rank-toleranssia
        max(eig) * n * eps. Aiempi versio tarjosi molemmat sb2_reference_moden
        takana; jäljelle jäi suhteellinen.
        """
        if vals.numel() == 0:
            return vals
        max_val = float(vals.max().item())
        if not math.isfinite(max_val) or max_val <= 0.0:
            return th.zeros_like(vals)
        threshold = max_val * float(vals.numel()) * float(th.finfo(vals.dtype).eps)
        return th.where(vals > threshold, vals, th.zeros_like(vals))

    def _mature_module_count(self) -> int:
        """Moduulit, joiden alkuakkumulointi on valmis.

        Refresh-yritys ilman yhtäkään kypsää moduulia ei tuota hajotelmaa, mutta
        se on ajoitustila eikä eigensolverin virhe. Sen laskeminen virheeksi sai
        _MAX_REFRESH_FAILURES keskeyttämään terveen ajon väärällä viestillä.
        """
        return sum(
            1
            for module in self.modules
            if self.module_states[module].A is not None
            and self.module_states[module].G is not None
            and int(self.module_states[module].stats_count) >= self.stats_accum_iter
        )

    @th.no_grad()
    def _update_inverses(self) -> int:
        """Päivitä kaikki saatavilla olevat spektrifaktorit atomisesti.

        Jokainen hajotelma lavastetaan ensin. Backendin virhe jättää edellisen
        faktorisukupolven koskemattomaksi; epääärellinen kovarianssi nollaa koko
        statistiikan, jotta vanhaa ja uutta sukupolvea ei voi sekoittaa.
        """
        staged: dict[
            nn.Module, tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor, str]
        ] = {}
        for module in self.modules:
            state = self.module_states[module]
            if state.A is None or state.G is None:
                continue
            # Sama portti kuin SB2:n stats_accum_iter. Moduulikohtaisena se
            # suojaa myös osittain palautetut faktorit liian aikaiselta käytöltä.
            if int(state.stats_count) < self.stats_accum_iter:
                continue
            try:
                expected_A, expected_G = self._expected_factor_shapes(module)
                if (
                    tuple(state.A.shape) != expected_A
                    or tuple(state.G.shape) != expected_G
                ):
                    raise RuntimeError(
                        f"Invalid K-FAC factor shape for {module}: "
                        f"A={tuple(state.A.shape)} expected={expected_A}, "
                        f"G={tuple(state.G.shape)} expected={expected_G}."
                    )
                a_vals, a_vecs, a_backend = self._safe_symmetric_eigh(
                    state.A, module=module, factor_name="A"
                )
                g_vals, g_vecs, g_backend = self._safe_symmetric_eigh(
                    state.G, module=module, factor_name="G"
                )
                self._require_finite(a_vals, f"A eigenvalues for {module}")
                self._require_finite(g_vals, f"G eigenvalues for {module}")
                self._check_large_finite(a_vecs, f"A eigenvectors for {module}")
                self._check_large_finite(g_vecs, f"G eigenvectors for {module}")
                staged[module] = (
                    self._clamp_eigenvalues(a_vals),
                    a_vecs,
                    self._clamp_eigenvalues(g_vals),
                    g_vecs,
                    f"A:{a_backend},G:{g_backend}",
                )
            except FloatingPointError as exc:
                self.eigh_failures += 1
                self._restart_kfac_state(clear_optimizer_state=True)
                raise FloatingPointError(
                    f"{exc} The K-FAC factor state was cleared and this exception "
                    "aborts the run. Catch it around learn() if you intend to "
                    "restart from the cleared state."
                ) from exc
            except Exception as exc:
                self.eigh_failures += 1
                self._warn_once(
                    f"Atomic K-FAC factor refresh failed for {module} "
                    f"({type(exc).__name__}: {exc}). Keeping the complete previous "
                    "factor generation."
                )
                return 0

        if not staged:
            return 0
        backends: list[str] = []
        for module, (a_vals, a_vecs, g_vals, g_vecs, backend) in staged.items():
            state = self.module_states[module]
            state.A_eigvals = a_vals
            state.A_eigvecs = a_vecs
            state.G_eigvals = g_vals
            state.G_eigvecs = g_vecs
            backends.append(backend)
        self.last_eigh_backend = ";".join(sorted(set(backends)))
        return len(staged)

    def _register_refresh_failure(self) -> None:
        """Keskeytä ajo, joka ei enää saa esikunnitinta rakennettua.

        Edellisen sukupolven säilyttäminen on oikea vastaus yksittäiseen
        ohimenevään virheeseen. Ikuisesti se ei ole: ajo muuttuisi hiljaisesti
        momentum-SGD:ksi ACKTR:n learning ratella, ja ainoa merkki olisi
        pysähtynyt train/kfac_factor_steps.
        """
        self.consecutive_refresh_failures += 1
        if self.consecutive_refresh_failures >= _MAX_REFRESH_FAILURES:
            raise RuntimeError(
                f"K-FAC factor refresh failed {self.consecutive_refresh_failures} "
                "times in a row, so no preconditioner is available and training "
                "would silently continue as plain momentum SGD. See the preceding "
                "warnings for the underlying eigensolver error."
            )
        self._warn_once(
            "K-FAC factor update produced no new valid decompositions; continuing "
            "with the previous factor generation."
        )

    # ------------------------------------------------------------------
    # Päivityksen rakentaminen
    # ------------------------------------------------------------------

    @classmethod
    @th.no_grad()
    def _grad_matrix(cls, module: nn.Module) -> Optional[th.Tensor]:
        if module.weight.grad is None:
            return None
        grad = module.weight.grad
        if isinstance(module, nn.Conv2d):
            grad = grad.reshape(grad.shape[0], -1)
        if cls._uses_trainable_bias(module):
            if module.bias.grad is None:
                return None
            grad = th.cat([grad, module.bias.grad.unsqueeze(1)], dim=1)
        return grad

    @th.no_grad()
    def _momentum_candidate(
        self, state_key: Any, update: th.Tensor, buffer_name: str
    ) -> tuple[th.Tensor, Optional[tuple[Any, str, th.Tensor]]]:
        """Palauta äärellinen momentum-ehdokas muuttamatta optimizerin tilaa."""
        self._require_finite(update, f"{buffer_name} input update")
        if self.momentum <= 0.0:
            return update.clone(), None
        buffer = self.state.get(state_key, {}).get(buffer_name)
        if (
            buffer is None
            or buffer.shape != update.shape
            or buffer.device != update.device
            or buffer.dtype != update.dtype
        ):
            base = th.zeros_like(update)
        else:
            self._require_finite(buffer, f"existing {buffer_name}")
            base = buffer
        candidate = base.mul(self.momentum).add(update)
        self._require_finite(candidate, f"candidate {buffer_name}")
        return candidate, (state_key, buffer_name, candidate)

    @th.no_grad()
    def _natural_gradient(
        self, module: nn.Module, grad_matrix: th.Tensor
    ) -> Optional[th.Tensor]:
        state = self.module_states[module]
        if (
            state.A_eigvals is None
            or state.G_eigvals is None
            or state.A_eigvecs is None
            or state.G_eigvecs is None
        ):
            return None

        denom = (
            state.G_eigvals.unsqueeze(1) * state.A_eigvals.unsqueeze(0) + self.damping
        )
        self._require_finite(grad_matrix, f"gradient matrix for {module}")
        self._require_finite(denom, f"K-FAC denominator for {module}")
        if th.any(denom <= 0):
            raise FloatingPointError(f"Non-positive K-FAC denominator for {module}.")

        # Tarkka ratkaisu Kronecker-approksimaatiolle additiivisella dampingilla:
        # vec(update) = (A kron G + damping I)^-1 vec(grad).
        grad_eigen = state.G_eigvecs.t().matmul(grad_matrix).matmul(state.A_eigvecs)
        natural = state.G_eigvecs.matmul(grad_eigen / denom).matmul(
            state.A_eigvecs.t()
        )
        if not th.isfinite(natural).all():
            raise FloatingPointError(
                f"Non-finite K-FAC natural gradient for module {module}."
            )
        return natural

    @th.no_grad()
    def _plan_fallback_updates(
        self,
        exclude_ids: set[int],
        lr_scale: float,
        buffer_name: str,
        gradient_scale: float = 1.0,
    ) -> tuple[list[tuple[th.Tensor, th.Tensor]], list[tuple[Any, str, th.Tensor]]]:
        parameter_plans: list[tuple[th.Tensor, th.Tensor]] = []
        buffer_plans: list[tuple[Any, str, th.Tensor]] = []
        for group in self.param_groups:
            lr = float(group["lr"]) * lr_scale
            if not math.isfinite(lr) or lr < 0.0:
                raise FloatingPointError(f"Invalid fallback learning rate: {lr}.")
            for param in group["params"]:
                if param.grad is None or id(param) in exclude_ids:
                    continue
                self._require_finite(param.grad, "fallback gradient")
                velocity, buffer_plan = self._momentum_candidate(
                    param, param.grad * gradient_scale, buffer_name
                )
                candidate = param - lr * velocity
                self._require_finite(candidate, "candidate fallback parameter")
                parameter_plans.append((param, candidate))
                if buffer_plan is not None:
                    buffer_plans.append(buffer_plan)
        return parameter_plans, buffer_plans

    @th.no_grad()
    def _commit_update_plans(
        self,
        parameter_plans: list[tuple[th.Tensor, th.Tensor]],
        buffer_plans: list[tuple[Any, str, th.Tensor]],
    ) -> None:
        """Kirjoita esivalidoidut momentum- ja parametriehdokkaat atomisesti."""
        seen_parameters: set[int] = set()
        seen_buffers: set[tuple[int, str]] = set()
        for param, candidate in parameter_plans:
            if id(param) in seen_parameters:
                raise RuntimeError("The same parameter was scheduled for update twice.")
            seen_parameters.add(id(param))
            self._require_finite(candidate, "candidate parameter before commit")
        for state_key, name, candidate in buffer_plans:
            identity = (id(state_key), name)
            if identity in seen_buffers:
                raise RuntimeError("The same momentum buffer was scheduled twice.")
            seen_buffers.add(identity)
            self._require_finite(candidate, f"candidate {name} before commit")
        for state_key, name, candidate in buffer_plans:
            self.state[state_key][name] = candidate
        for param, candidate in parameter_plans:
            param.copy_(candidate)

    def next_step_is_cold(self) -> bool:
        # SB2: tf.cond(sgd_step > cold_iter, warm, cold). Laskuri alkaa nollasta,
        # joten cold_iter=10 tuottaa 11 kylmää päivitystä (arvot 0..10).
        return self.steps <= self.cold_iter

    def preconditioner_is_active(self) -> bool:
        return self.factor_steps > 0

    @th.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with th.enable_grad():
                loss = closure()

        next_step = self.steps + 1
        lr = float(self.param_groups[0]["lr"])
        if not math.isfinite(lr) or lr < 0.0:
            raise FloatingPointError(f"Invalid optimizer learning rate: {lr}.")

        if next_step <= self.cold_iter + 1:
            parameter_plans, buffer_plans = self._plan_fallback_updates(
                exclude_ids=set(), lr_scale=1.0, buffer_name="cold_momentum_buffer"
            )
            self._commit_update_plans(parameter_plans, buffer_plans)
            self.steps = next_step
            return loss

        stats_ready = self.stats_steps >= self.stats_accum_iter
        pending_factor_increment = 0
        if stats_ready:
            # VALINTA: väli mitataan edellisestä ONNISTUNEESTA päivityksestä.
            # SB2:n globaali modulo (stats_step % kfac_update) riippuu siitä,
            # mihin kohtaan laskuri sattui lämpimän vaiheen alkaessa.
            refresh_due = self._steps_since_refresh >= self.kfac_update
            if refresh_due and self._mature_module_count() == 0:
                refresh_due = False
            if refresh_due:
                if self._update_inverses() > 0:
                    pending_factor_increment = 1
                    self._steps_since_refresh = 0
                    self.consecutive_refresh_failures = 0
                else:
                    self._register_refresh_failure()
            # Laskuri kasvaa jokaisella kypsällä lämpimällä askeleella, myös
            # sillä joka juuri päivitti spektrin. Aiemmin onnistuneen päivityksen
            # askel ohitti kasvatuksen, jolloin kfac_update=1 päivitti spektrin
            # vain joka toinen askel.
            self._steps_since_refresh += 1

        use_preconditioner = (self.factor_steps + pending_factor_increment) > 0
        module_updates: list[tuple[nn.Module, th.Tensor]] = []
        represented_param_ids: set[int] = set()
        first_param = self.param_groups[0]["params"][0]
        v_f_v_unscaled = th.zeros((), device=first_param.device, dtype=first_param.dtype)

        for module in self.modules:
            grad = self._grad_matrix(module)
            if grad is None:
                continue
            self._require_finite(grad, f"gradient matrix for {module}")
            natural = self._natural_gradient(module, grad) if use_preconditioner else None
            direction = grad if natural is None else natural
            self._require_finite(direction, f"optimizer direction for {module}")
            if use_preconditioner:
                v_f_v_unscaled += (grad * direction).sum()
            represented_param_ids.add(id(module.weight))
            if self._uses_trainable_bias(module):
                represented_param_ids.add(id(module.bias))
            module_updates.append((module, direction))

        if use_preconditioner:
            for group in self.param_groups:
                for param in group["params"]:
                    if param.grad is None or id(param) in represented_param_ids:
                        continue
                    self._require_finite(param.grad, "unfactored gradient")
                    v_f_v_unscaled += (param.grad * param.grad).sum()

        self._require_finite(v_f_v_unscaled, "K-FAC v^T F v")
        raw_curvature = float(v_f_v_unscaled.item())
        negative_tolerance = max(self.eps, 1e-7 * max(1.0, abs(raw_curvature)))
        if use_preconditioner and raw_curvature < -negative_tolerance:
            raise FloatingPointError(
                f"K-FAC curvature product must be non-negative, got {raw_curvature}."
            )
        if raw_curvature < 0.0:
            v_f_v_unscaled = th.zeros_like(v_f_v_unscaled)
            raw_curvature = 0.0

        if use_preconditioner and raw_curvature > 0.0:
            v_f_v = v_f_v_unscaled * (lr * lr)
            self._require_finite(v_f_v, "learning-rate-scaled K-FAC v^T F v")
            scale = th.sqrt(self.kfac_clip / (v_f_v + self.eps))
            self._require_finite(scale, "K-FAC KL scale")
            scale_float = min(1.0, float(scale.item()))
        else:
            v_f_v = th.zeros_like(v_f_v_unscaled)
            scale_float = 1.0

        # SB2 rakentaa lämpimän haaran MomentumOptimizerilla, jonka learning rate
        # on lr*(1-momentum).
        warm_lr = lr * (1.0 - self.momentum)
        if not math.isfinite(warm_lr) or warm_lr < 0.0:
            raise FloatingPointError(f"Invalid warm learning rate: {warm_lr}.")

        parameter_plans: list[tuple[th.Tensor, th.Tensor]] = []
        buffer_plans: list[tuple[Any, str, th.Tensor]] = []
        updated_param_ids: set[int] = set()

        for module, direction in module_updates:
            # Paino ja bias jakavat yhden [W|b]-muotoisen momentum-puskurin,
            # koska K-FAC-päivitys muodostetaan niille yhdessä.
            velocity, buffer_plan = self._momentum_candidate(
                module.weight, direction * scale_float, "warm_momentum_buffer"
            )
            uses_bias = self._uses_trainable_bias(module)
            if uses_bias:
                weight_update = velocity[:, :-1]
                bias_update = velocity[:, -1]
            else:
                weight_update = velocity
                bias_update = None
            if isinstance(module, nn.Conv2d):
                weight_update = weight_update.reshape_as(module.weight)

            candidate_weight = module.weight - warm_lr * weight_update
            self._require_finite(candidate_weight, f"candidate weight of {module}")
            parameter_plans.append((module.weight, candidate_weight))
            updated_param_ids.add(id(module.weight))
            if uses_bias and bias_update is not None:
                candidate_bias = module.bias - warm_lr * bias_update
                self._require_finite(candidate_bias, f"candidate bias of {module}")
                parameter_plans.append((module.bias, candidate_bias))
                updated_param_ids.add(id(module.bias))
            if buffer_plan is not None:
                buffer_plans.append(buffer_plan)

        fallback_parameters, fallback_buffers = self._plan_fallback_updates(
            exclude_ids=updated_param_ids,
            lr_scale=1.0 - self.momentum,
            buffer_name="warm_momentum_buffer",
            gradient_scale=scale_float,
        )
        parameter_plans.extend(fallback_parameters)
        buffer_plans.extend(fallback_buffers)
        self._commit_update_plans(parameter_plans, buffer_plans)

        self.steps = next_step
        self.factor_steps += pending_factor_increment
        self.last_v_f_v = float(v_f_v.item())
        self.last_kl_scale = scale_float
        return loss

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def state_dict(self):
        state_dict = super().state_dict()
        # Ominaisvektorimatriiseja ei talleteta: 3137x3137 float32 on ~39 MB
        # kappale, ja ne ovat puhdas funktio A:sta ja G:stä. Lataaja pitää raa'at
        # kovarianssit ja ajoittaa päivityksen ensimmäiselle lämpimälle askeleelle.
        state_dict["kfac_module_stats"] = [
            {
                "name": self.module_name(module),
                "weight_shape": tuple(module.weight.shape),
                "A": self.module_states[module].A,
                "G": self.module_states[module].G,
                "stats_count": self.module_states[module].stats_count,
            }
            for module in self.modules
        ]
        state_dict["kfac_steps"] = self.steps
        state_dict["kfac_stats_steps"] = self.stats_steps
        state_dict["kfac_stats_accum_iter"] = self.stats_accum_iter
        state_dict["kfac_eigh_failures"] = self.eigh_failures
        state_dict["kfac_last_v_f_v"] = self.last_v_f_v
        state_dict["kfac_last_kl_scale"] = self.last_kl_scale
        state_dict["kfac_audit_version"] = SB2_AUDIT_VERSION
        return state_dict

    def load_state_dict(self, state_dict):
        # Älä muuta kutsujan omistamaa sanakirjaa.
        state_dict = dict(state_dict)
        module_stats = state_dict.pop("kfac_module_stats", None)
        kfac_steps = int(state_dict.pop("kfac_steps", 0))
        kfac_stats_steps = int(state_dict.pop("kfac_stats_steps", 0))
        saved_accum_iter = state_dict.pop("kfac_stats_accum_iter", None)
        kfac_eigh_failures = int(state_dict.pop("kfac_eigh_failures", 0))
        last_v_f_v = float(state_dict.pop("kfac_last_v_f_v", 0.0))
        last_kl_scale = float(state_dict.pop("kfac_last_kl_scale", 1.0))
        state_dict.pop("kfac_audit_version", None)
        # Vanhojen versioiden kentät, joita tämä versio ei enää tunne.
        for legacy_key in (
            "kfac_factor_steps",
            "kfac_saved_eigen_state",
            "kfac_damping_scaling",
            "kfac_last_eigh_backend",
        ):
            state_dict.pop(legacy_key, None)
        super().load_state_dict(state_dict)

        self.steps = max(0, kfac_steps)
        self.eigh_failures = max(0, kfac_eigh_failures)
        self.consecutive_refresh_failures = 0
        self.last_eigh_backend = "none"
        self.last_v_f_v = last_v_f_v if math.isfinite(last_v_f_v) else 0.0
        self.last_kl_scale = (
            last_kl_scale
            if math.isfinite(last_kl_scale) and last_kl_scale >= 0.0
            else 1.0
        )

        if saved_accum_iter is not None and int(saved_accum_iter) != self.stats_accum_iter:
            warnings.warn(
                f"The checkpoint used stats_accum_iter={int(saved_accum_iter)} but "
                f"this optimizer uses {self.stats_accum_iter}; the restored "
                "per-module sample counts are interpreted against the current value.",
                UserWarning,
            )

        if module_stats is None or len(module_stats) != len(self.modules):
            if module_stats is not None or kfac_stats_steps:
                warnings.warn(
                    "Saved K-FAC module statistics are missing or do not match the "
                    "current model; statistics and counters were discarded.",
                    UserWarning,
                )
            self._restart_kfac_state(clear_optimizer_state=True)
            return

        staged: dict[nn.Module, KFACModuleState] = {}
        try:
            for module, saved in zip(self.modules, module_stats):
                expected_name = self.module_name(module)
                if saved.get("name", expected_name) != expected_name or tuple(
                    saved.get("weight_shape", tuple(module.weight.shape))
                ) != tuple(module.weight.shape):
                    raise ValueError(f"module identity mismatch for {expected_name}")

                device = module.weight.device
                dtype = module.weight.dtype

                def restore(key: str) -> Optional[th.Tensor]:
                    value = saved.get(key)
                    if value is None:
                        return None
                    result = value.to(device=device, dtype=dtype)
                    if not th.isfinite(result).all():
                        raise ValueError(f"non-finite checkpoint tensor {key}")
                    return result

                restored = KFACModuleState(
                    A=restore("A"),
                    G=restore("G"),
                    stats_count=max(0, int(saved.get("stats_count", 0))),
                )
                if (restored.A is None) != (restored.G is None):
                    raise ValueError(f"partial A/G factor state for {module}")
                expected_A, expected_G = self._expected_factor_shapes(module)
                for key, value, expected in (
                    ("A", restored.A, expected_A),
                    ("G", restored.G, expected_G),
                ):
                    if value is not None and tuple(value.shape) != expected:
                        raise ValueError(
                            f"{key} shape mismatch for {module}: "
                            f"{tuple(value.shape)} != {expected}"
                        )
                staged[module] = restored
        except Exception as exc:
            warnings.warn(
                f"Saved K-FAC state is incompatible or invalid ({type(exc).__name__}: "
                f"{exc}); statistics and counters were discarded.",
                UserWarning,
            )
            self._restart_kfac_state(clear_optimizer_state=True)
            return

        self.module_states.update(staged)
        self.stats_steps = max(0, kfac_stats_steps)
        # Ominaiskanta rakennetaan uudelleen ensimmäisellä lämpimällä askeleella.
        self.factor_steps = 0
        self._steps_since_refresh = self.kfac_update


class ACKTR(A2C):
    """SB2 ACKTR:n API ja ydinalgoritmi SB3:lle / PyTorchille.

    shared_trunk=True (oletus) antaa actorille ja criticille saman rungon;
    erillisiä ovat vain action_net- ja value_net-päät. shared_trunk=False
    rakentaa kaksi erillistä runkoa samalla net_archilla. Ks. moduulin alun
    selitys siitä, miksi tämä ei ole sama asia kuin SB3:n
    share_features_extractor.

    Runko määritellään aina eksplisiittisesti, kolmella argumentilla:

        havainto
          -> features_extractor_class (None = SB3:n oletus avaruuden mukaan)
          -> net_arch-mittainen MLP, aktivaationa activation_fn
          -> action_net / value_net

    Esimerkkejä:
        net_arch=(64, 64)                       SB2:n MlpPolicy
        net_arch=()                             päät suoraan extractorin päälle
        net_arch=(), features_extractor_class=NatureCNN
                                                SB2:n CnnPolicy
        net_arch=(128,), features_extractor_class=NatureCNN,
        features_extractor_kwargs=dict(features_dim=256)
                                                CNN 256 -> jaettu 128 -> päät
        net_arch=(64, 64), shared_trunk=False   kaksi erillistä 64-64-runkoa

    HUOM: net_arch on ACKTR:n argumentti, ei policy_kwargs-kenttä. net_arch,
    features_extractor_class, features_extractor_kwargs ja
    share_features_extractor policy_kwargsissa ovat virhe, jotta arkkitehtuuri
    päätetään täsmälleen yhdessä paikassa. Kuvahavainnoille oletus (64, 64)
    lisää MLP:n NatureCNN:n päälle; SB2:n asettelu on net_arch=().

    SB2-vastaavat parametrit: gamma, n_steps, ent_coef, vf_coef, vf_fisher_coef,
    learning_rate, max_grad_norm, kfac_clip, lr_schedule, kfac_update,
    gae_lambda, policy_kwargs, seed.

    Lisätty: MultiDiscrete-tuki (SB2 tukee vain Discretea ja Boxia).

    Optimizer-askelten määrästä: SB2:n _train_step ajaa train_op:n KERRAN per
    rollout. Sen `for _ in range(len(obs))` -silmukka kutsuu vain
    learning_rate_schedule.value():a, eli kelaa lr-aikataulua rollout-batchin
    verran eteenpäin. Tämä portti tekee saman.
    """

    # Kaikki kolme aliasta osoittavat samaan luokkaan: SB3:n vastineet eroavat
    # vain features_extractor_classin oletuksesta, jonka ACKTR asettaa aina itse.
    policy_aliases: Dict[str, Type[ActorCriticPolicy]] = {
        "MlpPolicy": ACKTRPolicy,
        "CnnPolicy": ACKTRPolicy,
        "MultiInputPolicy": ACKTRPolicy,
    }

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 0.25,
        lr_schedule: str = "linear",
        n_steps: int = 20,
        gamma: float = 0.99,
        # SB2:n gae_lambda=None valitsee ei-GAE-runnerin, jonka advantage on
        # returns - values. SB3:ssa gae_lambda=1.0 on algebrallisesti sama.
        # Huom. SB3 käsittelee episode_starts-lipun ja aikakatkaisujen
        # bootstrappauksen hieman eri tavalla kuin SB2:n discount_with_dones,
        # joten katkaistujen episodien rajoilla arvot voivat poiketa.
        gae_lambda: float = 1.0,
        ent_coef: float = 0.01,
        vf_coef: float = 0.25,
        vf_fisher_coef: float = 1.0,
        max_grad_norm: float = 0.5,
        # ---------------- Runko (aina eksplisiittinen) ----------------
        # Runko on: features_extractor_class -> net_arch-mittainen MLP -> päät.
        # net_arch=(64, 64) on SB2:n MlpPolicy. NatureCNN:lle ja muille
        # extractoreille, jotka ovat jo itsessään runko, käytä net_arch=().
        # features_extractor_class=None valitsee SB3:n oletuksen havaintoavaruuden
        # mukaan: FlattenExtractor, NatureCNN tai CombinedExtractor.
        # shared_trunk=False rakentaa saman rungon kahtena erillisenä kopiona,
        # yhden politiikalle ja yhden arvofunktiolle.
        net_arch: Sequence[int] = (64, 64),
        activation_fn: Type[nn.Module] = nn.Tanh,
        shared_trunk: bool = False,
        features_extractor_class: Optional[Type[BaseFeaturesExtractor]] = None,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        # SB2 ACKTR:n K-FAC-vakiot ovat näiden oletusarvoja.
        kfac_clip: float = 0.001,
        kfac_update: int = 1,
        kfac_damping: float = 0.01,
        kfac_stats_decay: float = 0.99,
        kfac_momentum: float = 0.9,
        cold_iter: int = 10,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        # Rollout-tason äärellisyystarkistukset pakottavat host-device-synkronoinnin
        # joka askeleella. Kokonaistappio tarkistetaan aina.
        debug_finite_checks: bool = False,
        _init_setup_model: bool = True,
    ):
        self.vf_fisher_coef = float(vf_fisher_coef)
        self.kfac_clip = float(kfac_clip)
        self.kfac_update = int(kfac_update)
        self.kfac_damping = float(kfac_damping)
        self.kfac_stats_decay = float(kfac_stats_decay)
        self.kfac_momentum = float(kfac_momentum)
        self.cold_iter = int(cold_iter)
        self.debug_finite_checks = bool(debug_finite_checks)
        self._warned_unvisited_kfac_modules = False

        # SB2:n legacy Scheduler etenee rollout-batchin verran per päivitys.
        self._sb2_lr_step = 0.0
        self._sb2_schedule_total_timesteps = 1
        self._sb2_schedule_initialized = False

        policy_kwargs = dict(policy_kwargs or {})
        reserved = {"observation_space", "action_space", "lr_schedule", "use_sde"}
        removed = sorted(reserved.intersection(policy_kwargs))
        for key in removed:
            policy_kwargs.pop(key, None)
        if removed:
            warnings.warn(
                "Removed SB3-reserved entries from policy_kwargs to avoid duplicate "
                f"policy constructor arguments: {removed}.",
                UserWarning,
            )

        # Runko määritellään ACKTR:n omilla argumenteilla, ei policy_kwargsin
        # kautta, jotta on täsmälleen yksi paikka jossa arkkitehtuuri päätetään.
        conflicting = sorted(
            key
            for key in (
                "net_arch",
                "share_features_extractor",
                "features_extractor_class",
                "features_extractor_kwargs",
            )
            if key in policy_kwargs
        )
        if conflicting:
            raise ValueError(
                f"Configure the trunk through the ACKTR arguments net_arch/"
                f"activation_fn/shared_trunk/features_extractor_class/"
                f"features_extractor_kwargs, not through policy_kwargs. "
                f"Got policy_kwargs entries: {conflicting}."
            )
        if net_arch is None:
            raise ValueError(
                "net_arch must be an explicit sequence of trunk layer widths; use "
                "net_arch=() to put the whole trunk in the features extractor."
            )

        self.shared_trunk = bool(shared_trunk)
        if (
            not self.shared_trunk
            and not isinstance(policy, str)
            and not issubclass(policy, ACKTRPolicy)
        ):
            warnings.warn(
                f"{policy.__name__} does not inherit from ACKTRPolicy, so "
                "share_features_extractor is missing from its "
                "_get_constructor_parameters(). ACKTR.save()/load() is unaffected, "
                "but policy.save() followed by a policy-level load would rebuild a "
                "shared-trunk policy and silently overwrite the actor trunk with the "
                "critic trunk. Subclass ACKTRPolicy to avoid this.",
                UserWarning,
            )
        # SB3 rakentaa features extractorin kerran (True) tai kahtena erillisenä
        # instanssina (False). Koska koko runko elää extractorissa, tämä lippu
        # kahdentaa nimenomaan rungon eikä vain havainnon esikäsittelyä.
        policy_kwargs["share_features_extractor"] = self.shared_trunk
        # SB3:n MlpExtractor rakentaisi net_archista kaksi erillistä haaraa, joten
        # se jätetään tyhjäksi ja runko elää features extractorissa.
        policy_kwargs["net_arch"] = []
        policy_kwargs["features_extractor_class"] = SharedTrunkExtractor
        policy_kwargs["features_extractor_kwargs"] = dict(
            trunk_arch=tuple(int(width) for width in net_arch),
            activation_fn=activation_fn,
            base_extractor_class=features_extractor_class,
            base_extractor_kwargs=dict(features_extractor_kwargs or {}),
        )

        super().__init__(
            policy=policy,
            env=env,
            learning_rate=make_sb2_lr_schedule(learning_rate, lr_schedule),
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            # SB3:n A2C loisi tässä RMSprop-optimizerin, jonka _setup_model
            # korvaa välittömästi K-FAC:lla; False välttää turhan
            # optimizer_kwargs-merkinnän policy_kwargsissa ja checkpointissa.
            use_rms_prop=False,
            use_sde=False,
            # SB2 normalisoi advantaget vain jatkuvassa action spacessa, mikä
            # tehdään train()-metodissa.
            normalize_advantage=False,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model,
        )

    # ------------------------------------------------------------------
    # Asetus
    # ------------------------------------------------------------------

    def _check_trunk_layout(self) -> None:
        """Varmista että toteutunut runkoasettelu vastaa pyydettyä.

        Lippu kulkee policylle policy_kwargsin kautta ja ACKTR.load() palauttaa
        policy_kwargsin checkpointista, joten ristiriita on mahdollinen ja olisi
        muuten näkymätön: ajo jatkuisi väärällä arkkitehtuurilla.
        """
        shares = bool(getattr(self.policy, "share_features_extractor", False))
        if shares != self.shared_trunk:
            raise RuntimeError(
                f"shared_trunk={self.shared_trunk} but "
                f"{type(self.policy).__name__}.share_features_extractor={shares}; "
                "the policy was built with a different trunk layout."
            )

        pi_extractor = getattr(self.policy, "pi_features_extractor", None)
        vf_extractor = getattr(self.policy, "vf_features_extractor", None)
        if pi_extractor is None or vf_extractor is None:
            raise RuntimeError(
                "The policy exposes no pi/vf features extractors; this ACKTR port "
                "requires an SB3 ActorCriticPolicy."
            )
        if self.shared_trunk and pi_extractor is not vf_extractor:
            raise RuntimeError(
                "The policy holds two distinct features extractor objects, so the "
                "actor and the critic do not share one network."
            )
        if not self.shared_trunk and pi_extractor is vf_extractor:
            raise RuntimeError(
                "shared_trunk=False but the policy holds a single features extractor "
                "object shared by the actor and the critic."
            )

        mlp_extractor = getattr(self.policy, "mlp_extractor", None)
        if mlp_extractor is not None:
            split_params = sum(
                parameter.numel()
                for net_name in ("policy_net", "value_net")
                for parameter in getattr(mlp_extractor, net_name, nn.Identity()).parameters()
            )
            if split_params:
                # Tämä on se kohta, jonka pelkkä share_features_extractor
                # jättäisi huomaamatta: SB3:n MlpExtractor rakentaa aina kaksi
                # erillistä haaraa, joten net_arch on oltava tyhjä, muuten
                # shared_trunk=True kahdentaisi rungon silti.
                raise RuntimeError(
                    f"policy.mlp_extractor holds {split_params} trainable parameters "
                    "in separate policy_net/value_net branches. net_arch must be "
                    "empty; the trunk belongs in the features extractor."
                )

    def _setup_model(self) -> None:
        # Irrota edellisen K-FAC-optimizerin hookit ennen kuin policy korvataan,
        # muuten vanha optimizer pitää vanhan policyn hengissä hookkiensa kautta.
        previous_optimizer = getattr(getattr(self, "policy", None), "optimizer", None)
        if isinstance(previous_optimizer, SB2KFACOptimizer):
            previous_optimizer.remove_hooks()

        super()._setup_model()

        if not isinstance(
            self.action_space, (spaces.Discrete, spaces.Box, spaces.MultiDiscrete)
        ):
            raise NotImplementedError(
                "This ACKTR port supports Discrete and Box action spaces like SB2, "
                f"plus MultiDiscrete. Got: {self.action_space}"
            )

        self._check_trunk_layout()

        if any(
            isinstance(module, (nn.RNNBase, nn.RNNCellBase))
            for module in self.policy.modules()
        ):
            warnings.warn(
                "The policy contains recurrent modules. This K-FAC port factors only "
                "Linear/Conv2d layers, so recurrent parameters silently fall back to "
                "the Euclidean momentum-SGD path and are not preconditioned.",
                UserWarning,
            )

        self.policy.optimizer = SB2KFACOptimizer(
            self.policy,
            lr=float(self.lr_schedule(1.0)),
            damping=self.kfac_damping,
            stats_decay=self.kfac_stats_decay,
            kfac_update=self.kfac_update,
            kfac_clip=self.kfac_clip,
            momentum=self.kfac_momentum,
            cold_iter=self.cold_iter,
            # SB2 aliasoi stats_accum_iter:in cold_iter:iin, koska ACKTR ei
            # koskaan aseta full_stats_init:iä.
            stats_accum_iter=self.cold_iter,
            debug_finite_checks=self.debug_finite_checks,
        )
        self._warned_unvisited_kfac_modules = False

    # ------------------------------------------------------------------
    # Koulutus
    # ------------------------------------------------------------------

    def _prepare_actions(self, actions: th.Tensor) -> th.Tensor:
        if isinstance(self.action_space, spaces.Discrete):
            return actions.long().flatten()
        if isinstance(self.action_space, spaces.MultiDiscrete):
            return actions.long()
        return actions

    def learn(
        self,
        total_timesteps: int,
        callback=None,
        log_interval: int = 100,
        tb_log_name: str = "ACKTR",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ):
        requested_timesteps = int(total_timesteps)
        if requested_timesteps < 1:
            raise ValueError("total_timesteps must be >= 1.")

        # VALINTA: SB2:n learn() rakentaa Scheduler(n_values=total_timesteps)
        # uudelleen joka kutsulla ja kelaa siis aina alkuun. Tässä kelataan vain
        # kun ajo todella aloitetaan alusta; muuten jatkettu learn()-kutsu
        # (reset_num_timesteps=False) hyppäisi lr_schedule="linear":lla takaisin
        # täyteen 0.25:een.
        if reset_num_timesteps or not self._sb2_schedule_initialized:
            self._sb2_lr_step = 0.0
            self._sb2_schedule_total_timesteps = requested_timesteps
            self._sb2_schedule_initialized = True
        else:
            self._sb2_schedule_total_timesteps += requested_timesteps

        return super().learn(
            total_timesteps=requested_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    def _set_sb2_learning_rate(self, step_increment: float) -> float:
        """Aseta lr SB2:n Scheduler-position mukaan.

        SB2 kutsuu Scheduler.value():a kerran per rolloutin ympäristönäytettä ja
        käyttää VIIMEKSI palautettua arvoa, eli positiota start + len(obs) - 1.
        Position lukeminen kohdasta start sijoittaisi jokaisen päivityksen
        kokonaisen rolloutin verran liian aikaisin käyrälle.
        """
        total = max(1, int(self._sb2_schedule_total_timesteps))
        self._sb2_lr_step += float(step_increment)
        position = max(0.0, float(self._sb2_lr_step) - 1.0)
        progress_remaining = max(0.0, 1.0 - position / float(total))
        lr = float(self.lr_schedule(progress_remaining))
        for param_group in self.policy.optimizer.param_groups:
            param_group["lr"] = lr
        return lr

    @staticmethod
    def _rollout_batch_size(observations: Any) -> int:
        if isinstance(observations, dict):
            if not observations:
                return 0
            return int(next(iter(observations.values())).shape[0])
        return int(observations.shape[0])

    def _require_training_finite(
        self, tensor: th.Tensor, description: str, *, force: bool = False
    ) -> None:
        """Äärellisyystarkistus, jonka kalliit muodot ovat oletuksena pois.

        Jokainen tarkistus pakottaa host-device-synkronoinnin, mikä hallitsee
        askeleen kestoa pienillä policyillä CUDA/MPS-laitteilla. Skalaarinen
        kokonaistappio tarkistetaan aina (yksi synkronointi, nappaa divergenssin).
        """
        if not (force or self.debug_finite_checks):
            return
        if not th.isfinite(tensor).all():
            raise FloatingPointError(f"Non-finite {description} in ACKTR training.")

    def train(self) -> None:
        self.policy.set_training_mode(True)
        optimizer = self.policy.optimizer
        # SB2 normalisoi advantaget vain jatkuville action spaceille.
        normalize_advantage = isinstance(self.action_space, spaces.Box)

        last_policy_loss = None
        last_value_loss = None
        last_entropy_loss = None
        last_fisher_loss = None
        last_lr = None
        optimizer_steps = 0

        # batch_size=None tuottaa koko rolloutin yhtenä eränä, joten tämä on
        # SB2:n yksi train_op per rollout.
        for rollout_data in self.rollout_buffer.get(batch_size=None):
            actions = self._prepare_actions(rollout_data.actions)
            batch_size = self._rollout_batch_size(rollout_data.observations)
            if batch_size <= 0:
                raise RuntimeError("ACKTR received an empty rollout batch.")

            advantages = rollout_data.advantages
            self._require_training_finite(advantages, "advantages")
            self._require_training_finite(rollout_data.returns, "returns")
            if normalize_advantage:
                advantages = (advantages - advantages.mean()) / (
                    advantages.std(correction=0) + 1e-8
                )

            # SB2 kelaa lr-aikataulua len(obs) askelta per rollout.
            last_lr = self._set_sb2_learning_rate(float(batch_size))
            is_cold = optimizer.next_step_is_cold()

            optimizer.zero_grad(set_to_none=True)
            # SB2 rakentaa _update_stats_op:n cold/warm-tf.cond-haarojen
            # ULKOPUOLELLA, joten statistiikka kerätään myös kylmillä askeleilla
            # ja faktorit ovat kypsiä kun lämmin vaihe alkaa.
            optimizer.begin_fisher_stats_collection()

            try:
                # Yksi forward-graafi palvelee sekä Fisher-passia että
                # actor-critic-lossia, kuten SB2:n TF-graafi.
                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations, actions
                )
                values = values.flatten()
                self._require_training_finite(values, "value predictions")
                self._require_training_finite(log_prob, "action log probabilities")

                # SB2: tf.gradients(pg_fisher_loss + vf_fisher_loss, params).
                # YKSI backward, jotta jaetun rungon kerros näkee summan
                # (g_pi + g_vf) ristitermeineen, aivan kuten SB2.
                pg_fisher_loss = log_prob.mean()
                sampled_values = values + th.randn_like(values)
                vf_fisher_loss = -self.vf_fisher_coef * th.mean(
                    th.pow(values - sampled_values.detach(), 2)
                )
                fisher_loss = pg_fisher_loss + vf_fisher_loss
                self._require_training_finite(fisher_loss, "joint Fisher loss")
                fisher_loss.backward(retain_graph=True)
                optimizer.complete_fisher_stats_collection()
                # Fisher-passin gradientit eivät saa vuotaa varsinaiseen
                # päivitykseen.
                optimizer.zero_grad(set_to_none=True)

                policy_loss = -(advantages * log_prob).mean()
                value_loss = F.mse_loss(rollout_data.returns, values)
                entropy_loss = log_prob.mean() if entropy is None else -entropy.mean()
                loss = (
                    policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
                )
                self._require_training_finite(policy_loss, "policy loss")
                self._require_training_finite(value_loss, "value loss")
                self._require_training_finite(entropy_loss, "entropy loss")
                # Yksi skalaarinen synkronointi per askel; riittävän halpa aina.
                self._require_training_finite(loss, "total loss", force=True)
                loss.backward()
                if self.debug_finite_checks:
                    for parameter in self.policy.parameters():
                        if parameter.grad is not None:
                            self._require_training_finite(parameter.grad, "gradient")

                # SB2 kutsuu clip_by_global_normia vain
                # KfacOptimizer._cold_sgd_startissa. Lämmin haara
                # (apply_gradients_kfac) ei leikkaa lainkaan: trust regionin
                # hoitaa vFv/kfac_clip-skaalaus.
                if self.max_grad_norm is not None and is_cold:
                    th.nn.utils.clip_grad_norm_(
                        self.policy.parameters(),
                        self.max_grad_norm,
                        error_if_nonfinite=self.debug_finite_checks,
                    )

                optimizer.step()
            except Exception:
                if optimizer.collecting_fisher_stats:
                    optimizer.discard_fisher_stats_collection()
                raise

            optimizer_steps += 1
            last_policy_loss = policy_loss.detach()
            last_value_loss = value_loss.detach()
            last_entropy_loss = entropy_loss.detach()
            last_fisher_loss = fisher_loss.detach()

        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(),
            self.rollout_buffer.returns.flatten(),
        )

        # Faktoroitu moduuli, joka ei koskaan saa Fisher-näytettä, ei tule
        # esikunnitetuksi. Raportoi kerran heti kun vertailukohtaa on.
        unvisited_modules: list[str] = []
        if optimizer.stats_steps > 0:
            unvisited_modules = optimizer.modules_without_statistics()
            if unvisited_modules and not self._warned_unvisited_kfac_modules:
                warnings.warn(
                    "These Linear/Conv2d modules received no Fisher statistics and "
                    "will not be preconditioned; they fall back to the Euclidean "
                    f"momentum path: {unvisited_modules}.",
                    UserWarning,
                )
                self._warned_unvisited_kfac_modules = True

        self._n_updates += 1
        record = self.logger.record
        record("train/n_updates", self._n_updates, exclude="tensorboard")
        record("train/explained_variance", explained_var)
        record("train/optimizer_steps", optimizer_steps)
        if last_lr is not None:
            record("train/learning_rate", last_lr)
        if last_policy_loss is not None:
            record("train/policy_loss", last_policy_loss.item())
        if last_value_loss is not None:
            record("train/value_loss", last_value_loss.item())
        if last_entropy_loss is not None:
            record("train/entropy_loss", last_entropy_loss.item())
        if last_fisher_loss is not None:
            record("train/fisher_loss", last_fisher_loss.item())
        record("train/kfac_v_f_v", optimizer.last_v_f_v)
        record("train/kfac_kl_scale", optimizer.last_kl_scale)
        record("train/kfac_stats_steps", optimizer.stats_steps)
        record("train/kfac_factor_steps", optimizer.factor_steps)
        record("train/kfac_eigh_failures", optimizer.eigh_failures)
        record("train/kfac_factored_modules", len(optimizer.modules))
        record("train/kfac_modules_without_stats", len(unvisited_modules))
        # Tärkein terveyssignaali: 0 tarkoittaa että ajo tekee momentum-SGD:tä
        # ACKTR:n learning ratella, ei K-FAC:ia.
        record(
            "train/kfac_preconditioner_active",
            int(optimizer.preconditioner_is_active()),
        )
        record(
            "train/kfac_consecutive_refresh_failures",
            optimizer.consecutive_refresh_failures,
        )
        if hasattr(self.policy, "log_std"):
            record("train/std", th.exp(self.policy.log_std).mean().item())


if __name__ == "__main__":
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.evaluation import evaluate_policy

    env = make_vec_env("CartPole-v1", n_envs=4, seed=0)

    model = ACKTR(
        "MlpPolicy",
        env,
        learning_rate=0.25,
        lr_schedule="linear",
        n_steps=20,
        gamma=0.99,
        ent_coef=0.01,
        vf_coef=0.25,
        vf_fisher_coef=1.0,
        max_grad_norm=0.5,
        # Runko: FlattenExtractor -> Linear(4,64)-Tanh-Linear(64,64)-Tanh, jonka
        # päällä ovat action_net ja value_net. shared_trunk=True tekee tästä
        # SB2:n MlpPolicyn; False rakentaisi rungon kahtena erillisenä kopiona.
        net_arch=(64, 64),
        activation_fn=nn.Tanh,
        shared_trunk=True,
        kfac_clip=0.001,
        kfac_update=10,
        kfac_damping=0.01,
        cold_iter=10,
        verbose=1,
        seed=0,
    )

    model.learn(total_timesteps=20_000)

    eval_env = make_vec_env("CartPole-v1", n_envs=1, seed=123)
    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=10, deterministic=True, warn=False
    )
    print(f"ACKTR mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")

    env.close()
