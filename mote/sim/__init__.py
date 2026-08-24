"""Verifiable-world data generator (grilled + signed 2026-08-24, docs/shape.md).

A tiny ECS simulates four micro-domains; one trace renders into three outputs (narrative doc,
QA pairs, DPO pairs) in three locales (en/ru/ja). Answers are read off the true component state,
so correctness holds by construction. The sim core steps independently of rendering and takes
agent actions as input — the future agency/RL environment reuses it unchanged.
"""
