import cv2
import streamlit as st
from ultralytics import YOLO
import base64

def app():
    st.header("Object Detection Web App")
    st.subheader("Powered by YOLOV8")
    st.write("Welcome!")
    model = YOLO("yolov8n.pt")
    objects = list(model.names.values())
    
    with st.form("my_form"):
        uploaded_file = st.file_uploader("Upload video", type=['mp4'])
        selected_objects = st.multiselect('Choose objects to detect from', objects, default=['person'])
        st.form_submit_button(label="Submit")
        confidence = st.slider('Confidence score', 0.0, 1.0)
        
        if uploaded_file is not None:
            input_file = uploaded_file.name
            bin_file = uploaded_file.read()
            with open(input_file, "wb") as temp:
                temp.write(bin_file)
            video = cv2.VideoCapture(input_file)
            width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(video.get(cv2.CAP_PROP_FPS))
            fourcc = cv2.VideoWriter_fourcc(*'X264') 
            output_dir = "output_videos"  
            os.makedirs(output_dir, exist_ok=True) 
            output_path = os.path.join(output_dir, f"{input_file.split('.')[0]}_output.mp4")
            out_video = cv2.VideoWriter(output_path, int(fourcc), fps, (width, height))

            with st.spinner("Processing video.."):
                while True:
                    ret, frame = video.read()
                    if not ret:
                        break
                    result = model(frame)
                    for detect in result[0].boxes.data:
                        x0, y0 = int(detect[0]), int(detect[1])
                        x1, y1 = int(detect[2]), int(detect[3])
                        confidence_score = round(float(detect[4]), 2)
                        classes = int(detect[5])
                        object_name = model.names[classes]
                        label = f'{object_name} {confidence_score}'
                        if object_name in selected_objects and confidence_score > confidence:
                            cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 0, 0), 2)
                            cv2.putText(frame, label, (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    detection = result[0].verbose()
                    cv2.putText(frame, detection, (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    out_video.write(frame)
                video.release()
                out_video.release()
            
            st.markdown(get_binary_file_downloader_html(output_path, 'Processed Video Download'), unsafe_allow_html=True)

def get_binary_file_downloader_html(bin_file, file_label='File'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:video/mp4;base64,{bin_str}" download="{file_label}.mp4">Download {file_label}</a>'
    return href

if __name__ == "__main__":
    app()
