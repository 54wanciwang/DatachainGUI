import streamlit as st

st.set_page_config(
    page_title="腾宇悦目标检测",       # 浏览器标签页上显示的文字
    page_icon="🚀",                # favicon：可以是 emoji，也可以是本地图片文件路径
    layout="wide",                   # （可选）页面布局方式
    initial_sidebar_state="expanded" # （可选）侧边栏默认展开
)

from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import tempfile
import time
import os, sys

# ✅ 检测是否是 PyInstaller 打包环境




BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATHS = {
    "Datachain_交通目标识别_light": os.path.join(BASE_DIR, "models", "datachain_o.pt"),
    "Datachain_交通目标识别_base": os.path.join(BASE_DIR, "models", "datachain_light.pt"),
    "Datachain_交通目标识别_large": os.path.join(BASE_DIR, "models", "datachain_max.pt"),
}
# MODEL_PATHS = {
#     "base": os.path.join(BASE_DIR, "models", "datachain_o.pt"),
#     "base": os.path.join(BASE_DIR, "models", "datachain_o.pt"),
# }

model_name = st.sidebar.selectbox("选择模型", list(MODEL_PATHS.keys()))
model = YOLO(MODEL_PATHS[model_name])

# ---------------------
# 推理函数
# ---------------------
def process_image(image):
    results = model(image)[0]
    for box in results.boxes:
        r = box.xyxy[0].cpu().numpy().astype(int)
        conf = float(box.conf)
        cls = int(box.cls)
        label = f'{model.names[cls]} {conf:.2f}'
        cv2.rectangle(image, (r[0], r[1]), (r[2], r[3]), (255, 0, 0), 2)
        cv2.putText(image, label, (r[0], r[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    return image

# ---------------------
# 前端结构
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
# 图像识别
# ---------------------
if option == "Image":
    uploaded_file = st.file_uploader("上传图像", type=['jpg', 'jpeg', 'png'])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)
        processed = process_image(image_np.copy())
        st.image(processed, caption="识别结果")

        if st.button("保存结果图像"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as f:
                Image.fromarray(processed).save(f.name)
                st.success(f"已保存至: {f.name}")


# ---------------------
# 视频识别
# ---------------------
elif option == "Video":
    uploaded_video = st.file_uploader("上传视频", type=['mp4', 'avi', 'mov'])
    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        cap = cv2.VideoCapture(tfile.name)

        stframe = st.empty()
        output_path = os.path.join(os.path.dirname(tfile.name), "output.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        size = (int(cap.get(3)), int(cap.get(4)))
        out = cv2.VideoWriter(output_path, fourcc, fps, size)

        progress = st.progress(0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed = process_image(frame.copy())
            out.write(cv2.cvtColor(processed, cv2.COLOR_RGB2BGR))
            stframe.image(processed, channels="RGB", caption=f"帧: {frame_idx}")
            frame_idx += 1
            progress.progress(min(frame_idx / total_frames, 1.0))

        cap.release()
        out.release()
        st.success(f"视频识别完成！已保存为: {output_path}")
