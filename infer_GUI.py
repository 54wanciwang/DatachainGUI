import streamlit as st
import io
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import tempfile
import time
import os, sys

# 页面配置：浏览器标签和 favicon
st.set_page_config(
    page_title="腾宇悦目标检测",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------
# 模型配置
# ---------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATHS = {
    "Datachain_交通目标识别_light": os.path.join(BASE_DIR, "models", "datachain_light.pt"),
    "Datachain_交通目标识别_base":  os.path.join(BASE_DIR, "models", "datachain_o.pt"),
    "Datachain_交通目标识别_large": os.path.join(BASE_DIR, "models", "datachain_max.pt"),
    "Coco_通用识别_light":           os.path.join(BASE_DIR, "models", "yolo11n.pt"),
}

model_name = st.sidebar.selectbox("选择模型", list(MODEL_PATHS.keys()))
model = YOLO(MODEL_PATHS[model_name])

# ---------------------
# 推理函数
# ---------------------
def process_image(image: np.ndarray) -> np.ndarray:
    results = model(image)[0]
    for box in results.boxes:
        # 解析框坐标、置信度、类别
        r    = box.xyxy[0].cpu().numpy().astype(int)
        conf = float(box.conf)
        cls  = int(box.cls)
        label = f'{model.names[cls]} {conf:.2f}'
        # 画框和标签
        cv2.rectangle(image, (r[0], r[1]), (r[2], r[3]), (255, 0, 0), 2)
        cv2.putText(image, label, (r[0], r[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    return image

# ---------------------
# 前端总体布局
# ---------------------
st.title("交通视频/图像目标检测系统")
st.markdown(
    "<div style='color: gray; font-size: 20px;'>来源：第一届“数据链杯”人工智能算法大赛 · 腾宇悦团队</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div style='color: gray; font-size: 20px;'> </div>",
    unsafe_allow_html=True
)

option = st.sidebar.radio("输入类型", ["Image", "Video"])

# ---------------------
# 图像识别模块
# ---------------------
if option == "Image":
    uploaded_file = st.file_uploader("上传图像", type=['jpg', 'jpeg', 'png'])
    if uploaded_file is not None:
        # 读取并处理
        image     = Image.open(uploaded_file).convert("RGB")
        image_np  = np.array(image)
        processed = process_image(image_np.copy())
        st.image(processed, caption="识别结果", use_container_width=True)

        # 下载结果图像
        buf = io.BytesIO()
        Image.fromarray(processed).save(buf, format="PNG")
        buf.seek(0)
        st.download_button(
            label="📥 下载结果图像",
            data=buf,
            file_name="result.png",
            mime="image/png"
        )

# ---------------------
# 视频识别模块
# ---------------------
elif option == "Video":
    uploaded_video = st.file_uploader("上传视频", type=['mp4', 'avi', 'mov'])
    if uploaded_video is not None:
        # 保存临时文件
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_video.name)[1])
        tfile.write(uploaded_video.read())
        cap = cv2.VideoCapture(tfile.name)

        # 准备输出视频
        output_path = os.path.join(os.path.dirname(tfile.name), "output.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps    = cap.get(cv2.CAP_PROP_FPS)
        size   = (int(cap.get(3)), int(cap.get(4)))
        out    = cv2.VideoWriter(output_path, fourcc, fps, size)

        stframe = st.empty()
        progress = st.progress(0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_idx = 0

        # 逐帧推理
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed = process_image(frame.copy())
            out.write(cv2.cvtColor(processed, cv2.COLOR_RGB2BGR))

            # 实时展示与进度
            stframe.image(processed, channels="RGB", caption=f"帧: {frame_idx}", use_container_width=True)
            frame_idx += 1
            progress.progress(min(frame_idx / total_frames, 1.0))

        cap.release()
        out.release()

        st.success("视频识别完成！")

