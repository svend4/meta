"""
hexboya — Тотальная система боя: Q6-модель

Источник: «Тотальная система боя», В.В. Крюков, ООО «ТОТАЛ»

Ключевое открытие:
  6 окон тела бойца (3 уровня × 2 стороны), каждое ОТКРЫТО/ЗАКРЫТО
  → 2^6 = 64 боевых состояния = Q6 (6-мерный гиперкуб)
  → Каждый глиф И-цзин = боевое состояние тела по Крюкову
"""
from .hexboya import BodyState, FigureEight, SphereSystem, CombatLaws, AnimalCycle, main

__all__ = ["BodyState", "FigureEight", "SphereSystem", "CombatLaws", "AnimalCycle", "main"]
