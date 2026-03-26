# Ensemble 方案研究记录

> 2026-03-26 对话成果，待下次固化到代码

## 最佳效果：31ms, 3 面板 + 12 细节 + 7 列表

### 五层方案（偷师组合）

```
Pass 0: 饱和度分离（偷自 disco_ui_mask）
  HSV S 通道 → Otsu INV → 场景像素置零

Pass 1: 粗检测 → 面板级（偷自 20 元素方案）
  TopHat(80) → Otsu → 中膨胀 (3,20)/(10,3)
  面积 > 1.5% 窗口 = 面板

Pass 2: 面板内细检测（偷自 filtered_gray TopHat）
  在每个面板内: TopHat(auto kernel) → Otsu → 小膨胀 (2,8)/(4,2)
  不加 min_area 过滤（保留所有 >10x8 的元素）

Pass 3: ListQuantize（直接调用 ListQuantizeStage）
  需要精确的 zone_rect（任务列表区域，不是整个面板）
  zone_rect 应该从 Pass 1 的面板 + Pass 2 的细节元素位置推断

Pass 4: 合并 + 分级
  去重：列表 item 内的细节元素不重复画
  分级：面积 >30000=主体(绿), >3000=次要(橙), 其他=辅助(紫)
  面板=黄色, 列表=青色
```

### 关键参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 饱和度 MORPH_CLOSE kernel | (10,10) | 填补 UI 面板内的间隙 |
| 粗 TopHat kernel | 80 | 面板级内容提取 |
| 粗膨胀 | (3,20)/(10,3) | 水平连接文字行，垂直连接标题+内容 |
| 面板面积阈值 | 1.5% 窗口面积 | 太小的不算面板 |
| 细 TopHat kernel | auto (panel_short_side / 4, clamp 15-40) | |
| 细膨胀 | (2,8)/(4,2) | 小幅连接，不合并太多 |
| ListQuantize zone | 需精确指定 | 不能用整个面板，会过度扩展 |
| inside() 去重 tolerance | 5px | |
| min_area 过滤 | 不加或 >= 500 | 加 500 会丢细节，不加会有噪声 |

### 已知问题

1. **ListQuantize 的 zone_rect 需要手动指定** — 应该从 Pass 1 面板 + Pass 2 细节自动推断
2. **min_area 两难** — 500 丢细节，不加有噪声。需要更智能的过滤（比如检查元素是否在面板内）
3. **inside() 去重可能误杀** — 需要更精确的重叠判断
4. **右侧"警官信息不全"区域** — 在内容热力图方案里能检测到但在当前 ensemble 里丢了
5. **底部工具栏** — 作为面板检测到了，但内部图标没有细检测

### 待做

- [ ] 把 ensemble 固化为 `EnsemblePipeline` 或一组 Stage
- [ ] 写测试锁住效果：微信 ≥50 元素，ScreenToGif ≥40 元素，Disco Elysium 面板=3 列表=6
- [ ] ListQuantize zone_rect 自动推断
- [ ] Phase 3 多帧累积验证
