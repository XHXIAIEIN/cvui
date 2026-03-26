# cvui 图层架构重构

## 核心改动

### DetectionContext 图层化（已完成）
- `layers: dict[str, ndarray]` 替代 `gray`/`binary` 单字段
- `zones: list[tuple]` 面板区域列表
- `gray`/`binary` 保留为 property 向后兼容

### Stage 改造原则
1. 每个 Stage **只读前面的层、只写自己的层**
2. 如果 `ctx.zones` 不为空，Stage **只在 zones 内处理**
3. 原图 `ctx.img` **永远不修改**

### 新增 Stage
- `ZoneDetectorStage`: 从 `ui_mask` 层的内容分布推断面板区域 → `ctx.zones`
- `CompositeStage`: 最后一个 Stage，合成所有图层输出最终结果

### 现有 Stage 改造
| Stage | 改动 |
|-------|------|
| GrayscaleStage | 写 `layers["gray"]`（已通过 property 兼容） |
| SaturationFilterStage | 写 `layers["ui_mask"]` + 修改 `layers["gray"]` |
| TopHatStage | 读 `layers["gray"]`，写 `layers["foreground"]` |
| OtsuStage | 读 `layers["foreground"]` or `layers["gray"]`，写 `layers["binary"]` |
| DilateStage | 读写 `layers["binary"]` |
| ConnectedComponentStage | 读 `layers["binary"]`，**只在 zones 内找**，写 `rects` |
| NestedStage | 读 `zones` + `rects`，对大框做子检测 |
| ClassifyStage | 读 `rects`，写 `classifications` |
| ChannelAnalysisStage | 读 `img`，写 `ui_states` |
| LayoutPatternStage | 读 `rects`，写 `ui_states["layout_pattern"]` |

### 预设流水线更新
```python
game_pipeline():
  SaturationFilter → ZoneDetector → Grayscale → TopHat → Otsu
  → Dilate → ConnectedComponent → RectFilter → Merge
  → Classify → LayoutPattern

standard_pipeline():
  Grayscale → TopHat → Otsu → Dilate → ConnectedComponent
  → RectFilter → Merge  (no zones, full image)

full_pipeline():
  Grayscale → TopHat → Otsu → Dilate → ConnectedComponent
  → RectFilter → Merge → Nested → Classify → ChannelAnalysis
  → LayoutPattern
```
