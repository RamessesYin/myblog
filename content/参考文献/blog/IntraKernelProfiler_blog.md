---
title: "Intra-Kernel Profiler (IKP)"
tags:
  - GitHub项目
  - GPU profiling
  - CUDA
  - NVBit
  - CUPTI
---

# Intra-Kernel Profiler (IKP)

## 🏷️ 元数据
- **作者**: yao-jz (Albert)
- **年份**: 2026（首次提交 2026-03-25）
- **License**: Apache 2.0
- **Stars**: 106
- **类型**: 开源 GitHub 项目（无对应学术论文）

## 项目主页

[https://github.com/yao-jz/intra-kernel-profiler](https://github.com/yao-jz/intra-kernel-profiler)

## 项目简介

IKP（Intra-Kernel Profiler）是一个面向 CUDA 内核的区域级性能剖析框架，通过三套互补后端（Trace Profiler、NVBit Region Profiler、CUPTI Collectors）将指令计数、内存流量、硬件计数器和 stall 原因等指标精细归因到用户命名的代码区域，并在自包含的 IKP Explorer HTML 仪表盘中可视化展示。

## 核心依赖

- **NVBit**: [https://github.com/NVlabs/NVBit](https://github.com/NVlabs/NVBit) — NVIDIA GPU 二进制插桩框架（MICRO 2019）
- **CUPTI**: CUDA Toolkit 随附的官方性能工具接口
- **Perfetto**: [https://ui.perfetto.dev](https://ui.perfetto.dev) — Chrome Trace JSON 可视化工具
