"""mesh/roi/ — ROI 提取功能组件包。

本包只保留功能组件（bfs 生长、cleanup 切割、extract 裁剪、registry 算法
注册、segmentation 分割等）。调度入口已外置：
  - 生产管线 run_roi_pipeline → commands/batch_process_all.py
  - 预标注/平台算法 extract_back_roi → mesh/roi_extract.py
"""
