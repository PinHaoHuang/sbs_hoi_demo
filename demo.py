import sys, os
sys.path.append('third_party/behave-dataset')
import streamlit as st
import cv2
import numpy as np
import tempfile
import time
from data.utils import load_intrinsics
from util import plot_image_with_bboxes_and_keypoints, filter_keypoints_and_links
from videoio import Uint16Reader

import subprocess
import zmq
import multiprocessing

import plotly.graph_objects as go

import argparse

parser = argparse.ArgumentParser(description='Streamlit App Arguments')
parser.add_argument('--bar', action="store_true")

st.session_state.args = parser.parse_args()

pwd_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))

def start_rtmpose3d():
    root_dir = os.path.join(pwd_dir, "third_party/mmpose/projects/rtmpose3d")
    subprocess.run((f"conda run -n  openmmlab PYTHONPATH={root_dir} " 
                    f"CUDA_VISIBLE_DEVICES=0 python {root_dir}/demo/my_demo.py "
                    f"{root_dir}/demo/rtmdet_m_640-8xb32_coco-person.py " 
                    "https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth "
                    f"{root_dir}/configs/rtmw3d-l_8xb64_cocktail14-384x288.py "
                    f"{pwd_dir}/checkpoints/rtmw3d-l_8xb64_cocktail14-384x288-794dbc78_20240626.pth"), shell=True)

def start_4dhuman():
    python_pth = os.path.join(pwd_dir, 'demo_4DHuman.py')
    subprocess.run(f"conda run -n 4D-humans python {python_pth}", shell=True)

if "zmq_initialized" not in st.session_state:
    p1 = multiprocessing.Process(target=start_rtmpose3d)
    p2 = multiprocessing.Process(target=start_4dhuman)

    p1.start()
    p2.start()



    # Set up ZeroMQ context and sockets
    context = zmq.Context()
    socket1 = context.socket(zmq.REQ)
    socket1.connect("tcp://localhost:5555")

    socket2 = context.socket(zmq.REQ)
    socket2.connect("tcp://localhost:5557")


    kinects = load_intrinsics(intrinsic_folder='/home/phuang/sv871514lx_data/BEHAVE/calibs/intrinsics', 
                                        kids=[0,1,2,3])
    
    st.session_state.zmq_context = context
    st.session_state.socket1 = socket1
    st.session_state.socket2 = socket2
    st.session_state.kinects = kinects
    st.session_state.zmq_initialized = True


def generate_3d_plot_plotly(x, y, z, color):

    # rand_indices = np.random.choice(color.shape[0], 4000, replace=False)
    # x = x[rand_indices]
    # y = y[rand_indices]
    # z = z[rand_indices]
    # color = color[rand_indices]

    color_hex = ['#%02x%02x%02x' % (int(r), int(g), int(b)) for r, g, b in color]

   
    # Create a scatter plot in 3D
    fig = go.Figure(data=[go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(
            size=4,
            color=color_hex,  
            # colorscale='Viridis',
            opacity=1
        )
    )])

    # fig.update_layout(
    #     scene=dict(
    #         xaxis_title="X Axis",
    #         yaxis_title="Y Axis",
    #         zaxis_title="Z Axis"
    #     ),
    #     title="Human Physical States"
    # )
    fig.update_layout(
        scene=dict(
            xaxis=dict(
                showgrid=False,
                title='',
                showticklabels=False,
                zeroline=False,
            ),
            yaxis=dict(
                showgrid=False,
                title='',
                showticklabels=False,
                zeroline=False,
            ),
            zaxis=dict(
                showgrid=False,
                title='',
                showticklabels=False,
                zeroline=False,
            ),
            camera=dict(
                eye=dict(x=1.5, y=-1.5, z=1.3)  # Adjust these values to change the initial view angle
            )
        ),
        title="Human Physical States"
    )
    
    return fig

def make_time_series_plot(values_array, title):

    time_series_fig = go.Figure()

    for i, joint_name in zip(st.session_state.target_joint_ids, st.session_state.target_joint_names):
        time_series_fig.add_trace(go.Scatter(
            x=st.session_state.times,
            y=values_array[:, i],
            mode='lines',
            name=joint_name
        ))

    # Fix the plot size
    time_series_fig.update_layout(
        width=400,   # Adjust width as needed
        height=500,  # Adjust height as needed
        xaxis_title='Time',
        yaxis_title='Value',
        title=title
    )

    return time_series_fig


def make_bar_plot(values, title, unit, x_range=[0, 2]):
    target_joint_ids = np.array(st.session_state.target_joint_ids)

    bar_fig = go.Figure()

    bar_fig.add_trace(go.Bar(
        x=values[target_joint_ids],
        y=st.session_state.target_joint_names,
        orientation='h'  # Makes the bars horizontal
    ))

    # Update layout
    bar_fig.update_layout(
        width=400,   # Adjust width as needed
        height=500,  # Adjust height as needed
        xaxis_title=f'Value ({unit})',
        yaxis_title='Joint',
        title=title,
        yaxis=dict(autorange="reversed"),  # Reverse the y-axis to have the first joint on top
        xaxis_range=x_range,
    )
    return bar_fig

# Placeholder processing function
def process_and_display_skeleton(frame_color, frame_depth):
    # Dummy processing function: simply return example keypoints and links

    frame_rgb = cv2.cvtColor(frame_color,cv2.COLOR_BGR2RGB)

    st.session_state.socket1.send_pyobj(frame_rgb)
    res = st.session_state.socket1.recv_pyobj()

    human_bbox = res['pred']['bbox'][0]

    skeleton_links = res['meta_info']['skeleton_links']

    # print(res['meta_info'])

    keypoints_2d = res['pred']['transformed_keypoints']
    keypoints_3d = res['pred']['keypoints']

    keypoints_2d, links = filter_keypoints_and_links(np.array(keypoints_2d), skeleton_links)

    left_hand_2d, right_hand_2d = keypoints_2d[9], keypoints_2d[10]


    # print('kp2d')
    # print(np.array(keypoints_2d).shape)

    # print(keypoints_3d)
    cam_K = st.session_state.kinects[1].calibration_matrix
    _cam_K = cam_K * st.session_state.resize_ratio
    _cam_K[2,2] = 1.0

    data_for_4dhumans = {
        'rgb_image' : frame_rgb,
        'depth_image': frame_depth / 1000.0,
        'human_bbox': human_bbox,
        'cam_K' : _cam_K,
        'left_hand_2d': np.array(left_hand_2d),
        'right_hand_2d': np.array(right_hand_2d)
    }

    st.session_state.socket2.send_pyobj(data_for_4dhumans)
    det_res = st.session_state.socket2.recv_pyobj()


    obj_boxes = det_res['boxes']
    # obj_classes = det_res['classes']
    # obj_ids = det_res['ids']

    human_verts = det_res['human_verts'][0]

    
    # human_verts[:, 0:2] = -human_verts[:, 0:2]

    human_kp3ds = det_res['human_kp3ds'][0]
    human_kp2ds = det_res['human_kp2ds'][0]

    human_mask = det_res['human_mask']

    human_smpl = det_res['smpl_params']

    human_pnts = det_res['human_pnts']
    other_pnts = det_res['other_pnts']
    contact_pnts_3d = det_res['contact_pnts']

    if (contact_pnts_3d is not None) and (other_pnts.shape[0] > 100):
    # if (other_pnts is not None) and (other_pnts.shape[0] > 50):
        contact_pnts_2d = (_cam_K @ contact_pnts_3d.T).T
        # contact_pnts_2d = (_cam_K @ human_pnts.T).T
        contact_pnts_2d = contact_pnts_2d[:, :2] / contact_pnts_2d[:, 2, None]
        contact_pnts_2d = np.unique(contact_pnts_2d.astype(int), axis=0)

    else:
        contact_pnts_2d = None

    # if other_pnts is not None:

    #     point_cloud1 = o3d.geometry.PointCloud()
    #     point_cloud2 = o3d.geometry.PointCloud()
    #     point_cloud1.points = o3d.utility.Vector3dVector(human_pnts)
    #     point_cloud1.colors = o3d.utility.Vector3dVector(np.array([[1,0,0]] * len(human_pnts)))
    #     point_cloud2.points = o3d.utility.Vector3dVector(other_pnts)
    #     point_cloud2.colors = o3d.utility.Vector3dVector(np.array([[0,0,1]] * len(other_pnts)))

    #     _, inliers = point_cloud2.segment_plane(distance_threshold=0.01, 
    #                                             ransac_n=3, 
    #                                             num_iterations=1000)
        
    #     point_cloud2 = point_cloud2.select_by_index(inliers, invert=True)

    #     if contact_pnts_3d.shape[0] > 0:
    #         point_cloud3 = o3d.geometry.PointCloud()
    #         point_cloud3.points = o3d.utility.Vector3dVector(contact_pnts_3d)
    #         point_cloud3.colors = o3d.utility.Vector3dVector(np.array([[0,1,0]] * len(contact_pnts_3d)))
    #         o3d.visualization.draw_geometries([point_cloud1, point_cloud2, point_cloud3])
    #     else:
    #         o3d.visualization.draw_geometries([point_cloud1, point_cloud2])
    # print(human_smpl[0]['global_orient'].shape, human_smpl[0]['body_pose'].shape, )

    theta = np.radians(-90)

    # Rotation matrix around the x-axis
    rotation_matrix = np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta),  np.cos(theta)]
    ])

    human_verts = human_verts @ rotation_matrix.T

    verts_x_med = np.median(human_verts[:, 0])
    verts_y_med = np.median(human_verts[:, 1])
    human_verts[:, 0] -= verts_x_med
    human_verts[:, 1] -= verts_y_med

    
    return {
        'keypoints_2d': keypoints_2d,
        'links': links,
        'human_mask': human_mask,
        'human_verts': human_verts,
        'contact_pnts_2d': contact_pnts_2d
        # 'human_verts': pnts,
    }




def display_frame(
                  frame_color=None, 
                  frame_depth=None, 
                  keypoints=None, 
                  links=None, 
                  masks=None, 
                  verts=None,
                  contact_pnts_2d=None,
                  values=None,
                  window_size=10,
                  smpl_color=None,
                  energy_values=None,
                  ):


    # frame_placeholder = st.session_state.frame_placeholder
    # plot_3d_placeholder = st.session_state.plot_3d_placeholder
    # time_series_placeholder = st.session_state.time_series_placeholder
    # time_series_placeholder1 = st.session_state.time_series_placeholder1
    

    if frame_color is None:

        for _, plot_placeholder in st.session_state.plot_placeholder_manager.items():
            if plot_placeholder['last_fig'] is not None:
                placeholder = plot_placeholder['placeholder']
                last_fig = plot_placeholder['last_fig']

                if plot_placeholder['type'] == 'image':
                    placeholder.image(last_fig, channels='BGR', use_container_width=True)  
                
                elif plot_placeholder['type'] == 'plot':
                    placeholder.plotly_chart(last_fig, use_container_width=True)
                
        return

        
    # Display the color frame in the first column
   
    if keypoints is not None:
        frame_color = plot_image_with_bboxes_and_keypoints(frame_color, keypoints=keypoints, links=links)


    # if contact_pnts_2d is not None:
       
    #     contact_pnts_2d = contact_pnts_2d.astype(int)
    #     for x,y in contact_pnts_2d:
    #         frame_color = cv2.circle(frame_color, (x, y), radius=3, color=(0,255,255), thickness=-1)
        # frame_color[contact_pnts_2d[:, 1], contact_pnts_2d[:, 0]] = np.array([0,255,255])

    frame_color = cv2.resize(frame_color, (640, 480))
            
    # Display the color frame in the first column
    st.session_state.plot_placeholder_manager['rgb_image']['placeholder'].image(frame_color, channels="BGR", use_container_width=True)
    st.session_state.plot_placeholder_manager['rgb_image']['last_fig'] = frame_color
    # frame_placeholder.image(frame_color, channels="BGR", use_container_width=True)

   
    if verts is not None:
        if smpl_color is None:
            smpl_color = np.array([[0.05098039, 0.03137255, 0.52941176]] * verts.shape[0])


        plot_3d_fig = generate_3d_plot_plotly(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2], color=smpl_color * 255)
        # st.session_state.last_plot_3d_fig = plot_3d_fig
        # plot_3d_placeholder.plotly_chart(plot_3d_fig, use_container_width=True)

        st.session_state.plot_placeholder_manager['smpl']['placeholder'].plotly_chart(plot_3d_fig, 
                                                                                     use_container_width=True,
                                                                                     key=f"smpl_{st.session_state.timestamp}")
        st.session_state.plot_placeholder_manager['smpl']['last_fig'] = plot_3d_fig


    if values is not None:
        if st.session_state.args.bar:
            values_fig = make_bar_plot(values, "Torque Values", "N-m", x_range=[0,30])

        else:

            # Append current time and values
            st.session_state.times.append(st.session_state.timestamp)
            st.session_state.values_over_time.append(values)

            # Keep only the last 'window_size' entries
            st.session_state.times = st.session_state.times[-window_size:]
            st.session_state.values_over_time = st.session_state.values_over_time[-window_size:]
        
            # Convert to numpy array for plotting
            values_array = np.array(st.session_state.values_over_time)  # Shape: (window_size, N)

            values_fig = make_time_series_plot(values_array, "Torque Values")
        st.session_state.plot_placeholder_manager['torque']['placeholder'].plotly_chart(values_fig, 
                                                                                        key=f'torque_{st.session_state.timestamp}')
        
        st.session_state.plot_placeholder_manager['torque']['last_fig'] = values_fig

       

    if energy_values is not None:

        if st.session_state.args.bar:
            energy_fig = make_bar_plot(energy_values, "Energy Values", "W", x_range=[0, 1])

        else:
            st.session_state.energy_times.append(st.session_state.timestamp)
            st.session_state.energy_over_time.append(energy_values)

            st.session_state.energy_times = st.session_state.energy_times[-window_size:]
            st.session_state.energy_over_time = st.session_state.energy_over_time[-window_size:]

            values_array = np.array(st.session_state.energy_over_time)

            energy_fig = make_time_series_plot(values_array, "Energy Values")

        st.session_state.plot_placeholder_manager['energy']['placeholder'].plotly_chart(energy_fig, 
                                                                                        key=f'energy_{st.session_state.timestamp}')
        
        st.session_state.plot_placeholder_manager['energy']['last_fig'] = energy_fig

        


# Function to load video and store initial configurations
# def load_video(video_file, resize_ratio, custom_fps):
#     if "video_capture" not in st.session_state:
#         # Initialize color video capture
#         cap_color = cv2.VideoCapture(video_file.name)
#         video_fps = cap_color.get(cv2.CAP_PROP_FPS)
#         st.session_state.video_capture = cap_color
#         st.session_state.video_fps = video_fps
#         st.session_state.total_frames = int(cap_color.get(cv2.CAP_PROP_FRAME_COUNT))
#         st.session_state.current_frame = 0
#         st.session_state.is_playing = False
#         st.session_state.resize_ratio = resize_ratio
#         st.session_state.custom_fps = custom_fps
#         st.session_state.skip_factor = max(1, int(video_fps / custom_fps))

#         # Initialize depth reader
#         depth_file_path = video_file.name.replace('.color.mp4', '.depth-reg.mp4')
#         st.session_state.depth_reader = Uint16Reader(depth_file_path)
#         st.session_state.depth_iter = iter(st.session_state.depth_reader)

#     # Adjust skip factor if custom_fps changes
#     if custom_fps != st.session_state.custom_fps:
#         st.session_state.custom_fps = custom_fps
#         st.session_state.skip_factor = max(1, int(st.session_state.video_fps / custom_fps))

def load_video(color_path, depth_path, resize_ratio, custom_fps):
    """
    Initializes or updates the video session state using file paths.
    """
    # Initialize only if the video is new or not yet loaded
    if "video_capture" not in st.session_state or st.session_state.get("color_path") != color_path:
        
        # Initialize color video capture using the correct path
        cap_color = cv2.VideoCapture(color_path)
        video_fps = cap_color.get(cv2.CAP_PROP_FPS)

        st.session_state.video_capture = cap_color
        st.session_state.video_fps = video_fps
        st.session_state.total_frames = int(cap_color.get(cv2.CAP_PROP_FRAME_COUNT))
        st.session_state.current_frame = 0
        st.session_state.is_playing = False
        st.session_state.resize_ratio = resize_ratio
        st.session_state.custom_fps = custom_fps
        st.session_state.skip_factor = max(1, int(video_fps / custom_fps))
        
        # Store the correct paths in session state for future checks
        st.session_state.color_path = color_path
        st.session_state.depth_path = depth_path

        # Initialize depth reader using the correct path
        st.session_state.depth_reader = Uint16Reader(depth_path)
        st.session_state.depth_iter = iter(st.session_state.depth_reader)
        
        st.info("Video players initialized successfully!")

    # Adjust skip factor if custom_fps changes on the fly
    if custom_fps != st.session_state.get("custom_fps"):
        st.session_state.custom_fps = custom_fps
        st.session_state.skip_factor = max(1, int(st.session_state.video_fps / custom_fps))

# Main video playback loop
def playback_loop():
    # frame_placeholder = st.empty()
    # plot_3d_placeholder = st.empty()
    col1, col2, col3, col4 = st.columns([3, 2, 2, 2])
    # st.session_state.frame_placeholder = col1.empty()
    # st.session_state.plot_3d_placeholder = col2.empty()
    # st.session_state.time_series_placeholder = col3.empty()
    # st.session_state.time_series_placeholder1 = col4.empty()

    st.session_state.plot_placeholder_manager = {
        'rgb_image': {'placeholder' : col1.empty(), 'type': 'image', 'last_fig': None},
        'smpl': {'placeholder' : col2.empty(), 'type': 'plot', 'last_fig': None},
        'torque': {'placeholder' : col3.empty(), 'type': 'plot', 'last_fig': None},
        'energy': {'placeholder' : col4.empty(), 'type': 'plot', 'last_fig': None},
        
    }

    st.session_state.times = []
    st.session_state.values_over_time = []

    st.session_state.energy_times = []
    st.session_state.energy_over_time = []
    


    while True:
        if st.session_state.is_playing:
            cap_color = st.session_state.video_capture

            # Set video position if resuming from a paused frame
            cap_color.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.current_frame)

            # Read the color frame
            ret_color, frame_color = cap_color.read()
            if not ret_color:
                break  # End of video

            # Read the corresponding depth frame
            try:
                frame_depth = next(st.session_state.depth_iter)
            except StopIteration:
                frame_depth = np.zeros_like(frame_color)  # Placeholder if no depth frame

            timestamp_sec = st.session_state.current_frame / st.session_state.video_fps

            st.session_state.timestamp = f'{timestamp_sec:.2f}'

            # Calculate seconds and milliseconds separately, rounding the milliseconds
            seconds = int(timestamp_sec)
            milliseconds = round((timestamp_sec - seconds) * 1000)
            
            # If milliseconds rounds up to 1000, adjust seconds and reset milliseconds to 0
            if milliseconds == 1000:
                seconds += 1
                milliseconds = 0

            # if seconds < 3: continue
            
            # Format the timestamp as 'tXXXX.XXX'
            timestamp_str = f"t{seconds:04}.{milliseconds:03}"
            
            # Format the timestamp as 'tXXXX.XXX' where XXXX are zero-padded seconds
            # timestamp_str = f"t{int(timestamp_sec):04}.{int((timestamp_sec % 1) * 1000):03}"

            # print(timestamp_str)

            if timestamp_str in st.session_state.torque_colors['joint_actuation_colors']:
                smpl_color = st.session_state.torque_colors['joint_actuation_colors'][timestamp_str][:, :3]

                # print(smpl_color.shape)
            else:
                smpl_color = None

            if timestamp_str in st.session_state.torque_colors['joint_actuation_values']:
                torque_values = st.session_state.torque_colors['joint_actuation_values'][timestamp_str].numpy()
                # print('hererere', torque_values.shape)
                torque_values = torque_values[0]

            else:
                
                torque_values = np.zeros(24)

            

            if timestamp_str in st.session_state.torque_colors['joint_energy']:
                joint_energy = st.session_state.torque_colors['joint_energy'][timestamp_str].numpy()
                # print('hereere', joint_energy.shape)
                joint_energy = joint_energy[0]
                
            
            else:
                
                joint_energy = np.zeros(24)


            # Resize frames based on resize_ratio
            h, w = frame_color.shape[:2]
            frame_color = cv2.resize(frame_color, (int(w * st.session_state.resize_ratio), int(h * st.session_state.resize_ratio)))
            frame_depth = cv2.resize(frame_depth, (int(w * st.session_state.resize_ratio), int(h * st.session_state.resize_ratio)))

            # Process and display
            st.session_state.model_start = time.time()
            if seconds < 3:
                result = None
            else:
                result = process_and_display_skeleton(frame_color, frame_depth)

            st.session_state.model_end = time.time()

            if result is not None:
                st.session_state.viz_start = time.time()
                display_frame(
                            #frame_placeholder, plot_3d_placeholder, 
                              frame_color, frame_depth, 
                              result['keypoints_2d'],
                              result['links'],
                              result['human_mask'],
                              result['human_verts'],
                              result['contact_pnts_2d'],
                              values=torque_values,
                              smpl_color=smpl_color,
                              energy_values=joint_energy)
                
                st.session_state.viz_end = time.time()

            elif seconds >= 3:
                display_frame(frame_color, frame_depth)

            

            # frame_color_placeholder, frame_depth_placeholder = frame_color, frame_depth
            # last_displayed_frame = (frame_color_placeholder, frame_depth_placeholder)

            # Move to the next frame according to the skip factor
            st.session_state.current_frame += st.session_state.skip_factor

            # Skip depth frames according to the skip factor
            for _ in range(st.session_state.skip_factor - 1):
                try:
                    next(st.session_state.depth_iter)
                except StopIteration:
                    break

            # Delay to match the custom FPS
            # time.sleep(1 / st.session_state.custom_fps)
        else:
            # When paused, display the last frame stored in session state
            # if "last_frame_color" in st.session_state and "last_frame_depth" in st.session_state:
            display_frame()
            time.sleep(0.1) 
            # print(frame_color_placeholder)
            # if frame_color_placeholder is not None:
            #     display_frame(frame_placeholder, frame_color_placeholder, frame_depth_placeholder)
            #     frame_color_placeholder, frame_depth_placeholder = None, None
            # time.sleep(0.1)  # If paused, wait without updating frames


st.set_page_config(layout="wide")
# Streamlit Setup
st.title("SBS Thrust 2 - Human Physical State Esimation with Object Interaction")


# Video upload and FPS control
# video_file = st.file_uploader("Upload a Video", type=["mp4"])
uploaded_files = st.file_uploader(
    "Upload both color and depth videos (.color.mp4 and .depth-reg.mp4)",
    type=["mp4"],
    accept_multiple_files=True
)
resize_ratio = st.slider("Resize Ratio", min_value=0.1, max_value=1.0, value=0.5)
custom_fps = st.slider("FPS", min_value=5, max_value=30, value=10)

st.session_state.torque_colors = np.load('data/estimated_torques_30fps_t0003.067_t0039.700.npy', allow_pickle=True).item() 


joint_names = [
    'pelvis',
    'left_hip',
    'right_hip',
    'spine1',
    'left_knee',
    'right_knee',
    'spine2',
    'left_ankle',
    'right_ankle',
    'spine3',
    'left_foot',
    'right_foot',
    'neck',
    'left_collar',
    'right_collar',
    'head',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
    'left_hand_finger',
    'right_hand_finger'
]

st.session_state.target_joint_names= [
    'spine1',
    'spine2',
    'spine3',
    'left_collar',
    'right_collar',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
    'left_hand_finger',
    'right_hand_finger'
]

st.session_state.target_joint_ids = [
   joint_names.index(name) for name in st.session_state.target_joint_names
]

# 2. Process files only if both have been uploaded
if uploaded_files and len(uploaded_files) == 2:
    color_file = None
    depth_file = None

    # Identify which file is which based on the name
    for file in uploaded_files:
        if ".color.mp4" in file.name:
            color_file = file
            color_base_name = file.name.replace(".color.mp4", "")
        elif ".depth-reg.mp4" in file.name:
            depth_file = file
            depth_base_name = file.name.replace(".depth-reg.mp4", "")

    # Ensure both files were identified correctly
    if color_file and depth_file and (color_base_name == depth_base_name):
        # 3. Save both files to temporary locations
        with tempfile.NamedTemporaryFile(delete=False, suffix=".color.mp4") as t_color, \
             tempfile.NamedTemporaryFile(delete=False, suffix=".depth-reg.mp4") as t_depth:
            
            t_color.write(color_file.getvalue())
            t_depth.write(depth_file.getvalue())

            # 4. Get the real, absolute paths of the temporary files
            color_path = os.path.abspath(t_color.name)
            depth_path = os.path.abspath(t_depth.name)

            st.success("Files saved to temporary paths:")
            st.code(f"Color Path: {color_path}\nDepth Path: {depth_path}", language=None)

            load_video(color_path, depth_path, resize_ratio, custom_fps)
            
            # Play/Pause button to toggle playback state
    if st.button("Play/Pause"):
        st.session_state.is_playing = not st.session_state.is_playing

    # Start the playback loop outside the button press
    playback_loop()


# Load video and initialize playback once
# if video_file:
#     load_video(video_file, resize_ratio, custom_fps)

#     # Play/Pause button to toggle playback state
#     if st.button("Play/Pause"):
#         st.session_state.is_playing = not st.session_state.is_playing

#     # Start the playback loop outside the button press
#     playback_loop()
