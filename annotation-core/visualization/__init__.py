"""可视化渲染层 — 纯 matplotlib 渲染，无计算无 I/O。

调用方从具体模块导入渲染函数（本文件仅为包级索引，不 re-export，
因 scatter_plot / paper_figures_panels 存在同名 render_scatter，扁平导出会歧义）。

按用途分组：

网格域（mesh / ROI 展示）
    cut_panels                render_mesh_panel / render_roi_extract_panel / render_single_panel ...
    landmarks_panels          render_curvature_landmarks_panel / render_waist_debug_panel
    parameterization_panels   draw_cut / draw_heightmap

评估图（预测 vs 真实）
    scatter_plot              render_scatter / render_scatter_3class / render_scatter_4class
    evaluation_panels         render_confusion_matrix_4class
    paper_figures_panels      render_scatter / render_bland_altman / render_confusion_matrix ...

导出图（瀑布图 / 特征重要性 / 轮廓）
    waterfall_panels          render_waterfall / render_residual_convergence / render_tree_structure
    feature_importance_panels render_top15_barh / render_importance_pie
    lateral_profile           plot_lateral_profile
"""
