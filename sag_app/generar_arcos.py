"""
Generador de arcos de la red G_i (Formulación v7).

Referencias al documento:
    - Edad del arco Δ_ij: Ecs. (1)-(2), Sección 1.3.
    - Cota de admisibilidad Δ_ij ≤ L_i + R_i: Ec. (6), Sección 3.4.
    - Definición de A_i: Ec. (7), Sección 3.4.
    - Cota de tamaño |A_i| ≤ (H+1)(L_i+R_i+1): Ec. (17), Sección 6.1.

Para cada sección i, un arco es el par j = (s, t) con:
    s ∈ {0} ∪ T      (nodo inicial: fuente o una semana)
    t ∈ T ∪ {∞}      (nodo final: una semana o el sumidero)
    s < t

Este módulo trae DOS generadores:

1. `generar_arcos_completos_seccion` / `generar_todos_los_arcos_completos`
   Generan TODOS los arcos topológicamente posibles, sin aplicar la cota
   Δ_ij ≤ L_i+R_i, porque R_i sigue sin dato (†, Sección 7.2). La única
   condición es s < t. Es el universo completo de arcos para las 9
   secciones y las H=156 semanas.

2. `generar_arcos_seccion` / `generar_todos_los_arcos`
   Aplican además la cota de admisibilidad Ec. (6), para cuando R_i (y
   L_i, e_i) tengan valores reales asignados. Se dejan listas para ese
   momento; no se usan en el ejemplo principal de este archivo.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from typing import Iterable, Union

# Sumidero ∞: se representa con float('inf') para que las comparaciones
# s < t y los cálculos de edad funcionen sin casos especiales.
INF = math.inf

Nodo = Union[int, float]  # int para semanas de T (y la fuente 0), float('inf') para el sumidero


@dataclass(frozen=True)
class Arc:
    """
    Un arco j = (s, t) de la red G_i de la sección i.

    delta (Δ_ij) es None cuando el arco se generó sin conocer e_i (universo
    completo, sin poda): la existencia del arco no depende de e_i, solo su
    edad. Se completa apenas e_i esté disponible.
    """
    i: int
    s: int
    t: Nodo
    delta: float | None = None

    @property
    def es_terminal(self) -> bool:
        return math.isinf(self.t)

    def __repr__(self) -> str:
        t_str = "∞" if self.es_terminal else str(self.t)
        d_str = f"{self.delta:g}" if self.delta is not None else "?"
        return f"Arc(i={self.i}, s={self.s}, t={t_str}, Δ={d_str})"


def edad(e_i: float, H: int, s: int, t: Nodo) -> float:
    """
    Δ_ij según Ecs. (1)-(2).

        t ∈ T:  Δ = e_i + t   si s = 0
                Δ = t - s     si s ∈ T
        t = ∞:  Δ = e_i + H   si s = 0
                Δ = H - s     si s ∈ T
    """
    if math.isinf(t):
        return e_i + H if s == 0 else H - s
    return e_i + t if s == 0 else t - s


# ---------------------------------------------------------------------------
# Universo completo de arcos, sin la cota Δ_ij ≤ L_i+R_i (R_i pendiente).
# Única condición: s < t.
# ---------------------------------------------------------------------------

def generar_arcos_completos_seccion(
    i: int, H: int, e_i: float | None = None
) -> list[Arc]:
    """
    Genera TODOS los arcos posibles de la sección i: cada par (s, t) con
    s ∈ {0}∪T, t ∈ T∪{∞}, s < t. No depende de L_i ni R_i.

    Δ_ij (Ecs. 1-2) se calcula por tramo, según si depende de e_i o no:
        - s ∈ T (arco entre dos recambios dentro del horizonte): Δ = t-s
          (o H-s si t=∞). No depende de e_i: se calcula siempre.
        - s = 0 (arco de la pieza ya instalada al inicio): Δ = e_i+t
          (o e_i+H si t=∞). Depende de e_i: si no se entrega, delta queda
          en None (pendiente) en vez de calcularse mal.

    |A_i| = (H+1)(H+2)/2 para cualquier sección (no depende de i ni de e_i).
    De esos arcos, exactamente H+1 salen de la fuente (uno por cada t ∈ T,
    más el terminal) y son los que quedan pendientes sin e_i; el resto,
    (H+1)(H+2)/2 - (H+1) = H(H+1)/2, ya tiene Δ calculada.
    """
    arcos: list[Arc] = []

    # --- arcos desde la fuente (s = 0): Δ depende de e_i ---
    for t in range(1, H + 1):
        d = e_i + t if e_i is not None else None
        arcos.append(Arc(i, 0, t, d))
    d_inf = e_i + H if e_i is not None else None
    arcos.append(Arc(i, 0, INF, d_inf))

    # --- arcos desde cada semana s ∈ T: Δ NO depende de e_i ---
    for s in range(1, H + 1):
        for t in range(s + 1, H + 1):
            arcos.append(Arc(i, s, t, t - s))
        arcos.append(Arc(i, s, INF, H - s))

    return arcos


def generar_todos_los_arcos_completos(
    H: int, secciones: Iterable[int] | dict[int, float]
) -> dict[int, list[Arc]]:
    """
    Genera {i: A_i} completo (sin poda) para un conjunto de secciones.

    secciones: iterable de i (sin e_i conocido) o dict {i: e_i} (con e_i).
    """
    items = secciones.items() if isinstance(secciones, dict) else (
        (i, None) for i in secciones
    )
    return {i: generar_arcos_completos_seccion(i, H, e_i) for i, e_i in items}


# ---------------------------------------------------------------------------
# Versión con poda: aplica además Δ_ij ≤ L_i+R_i (Ec. 6). Usar cuando R_i
# (y L_i, e_i) tengan valores reales asignados (Sección 7.2, pendiente).
# ---------------------------------------------------------------------------

def generar_arcos_seccion(
    i: int, e_i: float, L_i: float, R_i: float, H: int
) -> list[Arc]:
    """
    Genera A_i completo para la sección i (Ec. 7): todos los arcos
    admisibles desde la fuente 0 y desde cada semana s ∈ T, hacia semanas
    posteriores y hacia el sumidero ∞.
    """
    cota = L_i + R_i
    arcos: list[Arc] = []

    # --- arcos desde la fuente (s = 0) ---
    # t ∈ T:  e_i + t ≤ cota  →  t ≤ cota - e_i
    t_max = min(H, math.floor(cota - e_i))
    for t in range(1, t_max + 1):
        arcos.append(Arc(i, 0, t, edad(e_i, H, 0, t)))
    # t = ∞:  e_i + H ≤ cota
    if e_i + H <= cota:
        arcos.append(Arc(i, 0, INF, edad(e_i, H, 0, INF)))

    # --- arcos desde cada semana s ∈ T = {1, ..., H} ---
    for s in range(1, H + 1):
        # t ∈ T, t > s:  t - s ≤ cota  →  t ≤ s + cota
        t_max = min(H, s + math.floor(cota))
        for t in range(s + 1, t_max + 1):
            arcos.append(Arc(i, s, t, edad(e_i, H, s, t)))
        # t = ∞:  H - s ≤ cota
        if H - s <= cota:
            arcos.append(Arc(i, s, INF, edad(e_i, H, s, INF)))

    return arcos


def generar_todos_los_arcos(
    H: int, secciones: dict[int, dict[str, float]]
) -> dict[int, list[Arc]]:
    """
    Genera {i: A_i} para un conjunto de secciones N.

    secciones: {i: {"e_i": ..., "L_i": ..., "R_i": ...}, ...}
    """
    resultado: dict[int, list[Arc]] = {}
    for i, p in secciones.items():
        A_i = generar_arcos_seccion(i, p["e_i"], p["L_i"], p["R_i"], H)
        if not any(a.s == 0 for a in A_i):
            # Ningún arco sale de la fuente: la sección es infactible con
            # estos parámetros (pendiente "Factibilidad inicial", Sección 7.2).
            print(
                f"[AVISO] Sección {i}: ningún arco sale de la fuente 0 "
                f"(e_i={p['e_i']} > L_i+R_i={p['L_i']+p['R_i']}). "
                f"Modelo infactible para esta sección tal como está definida."
            )
        resultado[i] = A_i
    return resultado


# ---------------------------------------------------------------------------
# Índices A_i^-(t) y A_i^+(t) (arcos entrantes/salientes de un nodo),
# necesarios para las restricciones de conservación de flujo y acople
# (Ecs. 9-11, Sección 5.2). Se calculan una vez por sección.
# ---------------------------------------------------------------------------

def indices_entrada_salida(
    A_i: Iterable[Arc],
) -> tuple[dict[Nodo, list[Arc]], dict[Nodo, list[Arc]]]:
    """Retorna (A_i^-, A_i^+): dict nodo -> lista de arcos que entran / salen de él."""
    entrantes: dict[Nodo, list[Arc]] = {}
    salientes: dict[Nodo, list[Arc]] = {}
    for a in A_i:
        entrantes.setdefault(a.t, []).append(a)
        salientes.setdefault(a.s, []).append(a)
    return entrantes, salientes


# ---------------------------------------------------------------------------
# Exportación a CSV, útil para alimentar el resolvedor (una fila por arco).
# ---------------------------------------------------------------------------

def exportar_csv(arcos_por_seccion: dict[int, list[Arc]], path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["i", "s", "t", "delta"])
        for A_i in arcos_por_seccion.values():
            for a in A_i:
                t_out = "inf" if a.es_terminal else a.t
                d_out = a.delta if a.delta is not None else "pendiente_e_i"
                w.writerow([a.i, a.s, t_out, d_out])


# ---------------------------------------------------------------------------
# Ejemplo de uso: universo completo de arcos para N = {1,...,9}, H = 156.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    H = 156           # horizonte, Sección 1.1
    N = range(1, 10)  # 9 secciones, Sección 1.1

    arcos_por_seccion = generar_todos_los_arcos_completos(H, N)

    cota_teorica = (H + 1) * (H + 2) // 2  # |A_i| = C(H+2, 2), igual para toda i
    print(f"{'i':>3} {'|A_i|':>8}")
    total = 0
    for i, A_i in arcos_por_seccion.items():
        assert len(A_i) == cota_teorica, f"sección {i}: {len(A_i)} != {cota_teorica}"
        print(f"{i:>3} {len(A_i):>8}")
        total += len(A_i)

    print(f"\n|A_i| por sección: {cota_teorica}  (= (H+1)(H+2)/2, Ec. 17 sin cota)")
    print(f"Total de arcos, las 9 secciones × 156 semanas: {total}")

    # Desglose Δ calculada vs pendiente (por falta de e_i) para la sección 1
    A1 = arcos_por_seccion[1]
    con_delta = sum(1 for a in A1 if a.delta is not None)
    pendientes = sum(1 for a in A1 if a.delta is None)
    print(f"\nSección 1 — arcos con Δ calculada (s ∈ T): {con_delta}")
    print(f"Sección 1 — arcos pendientes de e_i (s = 0): {pendientes}")
    assert con_delta == H * (H + 1) // 2
    assert pendientes == H + 1

    # Índices A_i^-(t), A_i^+(t) para la sección 1 (necesarios para las
    # restricciones de flujo (9)-(11) una vez armado el MILP)
    entrantes, salientes = indices_entrada_salida(arcos_por_seccion[1])
    print(f"\nSección 1 — arcos que entran a la semana 40: {len(entrantes.get(40, []))}")
    print(f"Sección 1 — arcos que salen de la fuente 0: {len(salientes.get(0, []))}")

    exportar_csv(arcos_por_seccion, "arcos.csv")
    print("\nExportado a arcos.csv "
          "(delta = 'pendiente_e_i' en los arcos que salen de la fuente 0)")
