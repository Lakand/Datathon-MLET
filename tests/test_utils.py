# tests/test_utils.py
"""Testes Unitários de Funções Utilitárias.

Este módulo foca na validação das funções auxiliares e regras de negócio
isoladas contidas no módulo `src.utils`. Garante que a lógica de decisão
(como a classificação de risco) esteja alinhada com os requisitos do projeto.
"""

import pytest
from src.utils import calculate_risk_level

@pytest.mark.parametrize("pedra, risco_esperado", [
    ("Ametista", "Baixo"),
    ("Topázio", "Baixo"),
    ("Ágata", "Alto"),
    ("Quartzo", "Alto"),
    ("ametista", "Baixo"), # Testando a robustez do .capitalize()
    ("PEDRA DESCONHECIDA", "Alto") # Testando o caso default (else)
])
def test_regra_negocio_risco(pedra: str, risco_esperado: str):
    """Valida a regra de negócio de cálculo de risco (Pedra -> Risco).

    Utiliza `pytest.mark.parametrize` para testar diversas entradas contra
    as saídas esperadas, verificando:
    1. Se pedras de alto desempenho (Ametista, Topázio) retornam risco 'Baixo'.
    2. Se pedras de baixo desempenho (Ágata, Quartzo) retornam risco 'Alto'.
    3. Se a função normaliza corretamente a capitalização (ex: 'ametista').
    4. Se inputs desconhecidos caem na regra padrão de segurança (Risco Alto).

    Args:
        pedra (str): Nome da pedra (input simulado).
        risco_esperado (str): Classificação de risco esperada ('Baixo' ou 'Alto').
    """
    assert calculate_risk_level(pedra) == risco_esperado