# 手绘过程视频生成器

使用 `uv` 管理环境，将静态图片转换为手绘过程 MP4。画面从纸张底色开始，按构图与明暗、背景大色块、主体铺色、阴影与受光、线条和最终细节逐步推进；只有笔刷覆盖的位置会改变。颜色区域按面积近似分为背景和主体，因此通用图片上属于启发式分层。支持顺序或交错推进，并显示跟随当前笔尖的手、画笔和投影。

## 准备环境

需要 FFmpeg，并安装 uv。项目依赖由 `pyproject.toml` 和 `uv.lock` 管理：

```bash
uv sync
```

## 生成视频

```bash
uv run python painting_video.py test2.png output/test2_painting.mp4 \
  --duration 8 --fps 24 --effect sequential --brush watercolor
```

`--effect` 支持 `sequential`（顺序）和 `interleaved`（交错）；`--brush` 支持三种可见笔触：`pencil` 使用细线和颗粒留白，`watercolor` 使用较宽、柔边和半透明叠色，`marker` 使用较粗、边缘清晰且不透明的色带。最终收尾会完整补齐原图，不受铅笔颗粒影响。时长、帧率和最长边尺寸可通过 `--duration`、`--fps`、`--size` 调整。

默认使用 `assets/hand_brush.png` 中的实拍手部素材，并以照片里的笔尖为锚点跟随笔迹。竖屏时按笔尖在画面中轴的左右位置，让前臂从左侧或右侧出框；横屏时按上下位置，从顶部或底部出框。中轴附近有缓冲区，降低朝向抖动；手臂长度会自动保证末端越过画面边缘。可用 `--hand-scale 1.2` 放大手臂（范围 0.5–2.0，默认 1.0；为维持出框约束，过小值可能被自动抬高）。替换素材时可传 `--hand-image path/to/hand.png`（透明背景 PNG）；传 `--hand-image none` 可回退到程序绘制的手部。
