# Project Roadmap

This document outlines the planned development stages and key milestones for the project.

## Phase 1: Initial Setup and Basic Simulation (Completed)

-   Set up project structure and environment.
-   Implement basic linear state-space model.
-   Develop initial simulation environment.
-   Visualize basic simulation results.

## Phase 2: Optimization and Deployment (Future)

-   Optimize code for performance and efficiency.
-   Prepare for potential deployment.
-   Document the system and usage.
-   Plan for future enhancements and maintenance.

---

**Note:** This roadmap is subject to change based on project progress and findings.
*   Ensure all generated plots and visualizations use English labels, titles, and legends.
*   Python interpreter path: `/usr/local/bin/python3`
*   水平管中，我的电压是 0 伏到 10 伏，然后占空比就是从这个计算的，0 伏往左边吹，10 伏相反，然后最左边，质心是 0，最右边的时候，质心是 156mm
*   control input 的正确图片是 `12161745906913_.pic.jpg`
*   生成的 response 的正确趋势的图片是 `12151745906913_.pic.jpg`

*   閥芯位置 (s_spool) 隨時間變化的具體序列（數值和時間點）：  根据时间控制PWM输出
            if  elapsed_time <= 2.00:
                pwm_controller.set_duty_cycle(100, elapsed_time)
            elif elapsed_time <= 8.00:
                pwm_controller.set_duty_cycle(50, elapsed_time)
            elif elapsed_time <= 8.10:
                pwm_controller.set_duty_cycle(0, elapsed_time)
            else:
                pwm_controller.set_duty_cycle(50, elapsed_time)
