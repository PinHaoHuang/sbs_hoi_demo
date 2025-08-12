import cv2
import numpy as np
import open3d as o3d
import torch
from scipy.ndimage import label
from torch.profiler import profile, record_function, ProfilerActivity

def plot_image_with_bboxes_and_keypoints(image, bbox=None, keypoints=None, filter_ids=None, links=None):
    """
    Plots an image with bounding boxes (in xyxy format) and 2D keypoints (as offsets from the top-left corner of each box).
    Draws only the keypoints if their indices are in filter_ids and connects them based on the given links.

    Parameters:
    - image: A numpy array representing the image.
    - bboxes: A numpy array of shape (N, 4), where each row is (x1, y1, x2, y2).
    - keypoints: A numpy array of shape (N, K, 2), where each row contains K keypoints for a bounding box,
      and each keypoint is represented as (offset_x, offset_y) from the top-left corner of the bbox.
    - filter_ids: A list of indices indicating which keypoints to draw. If None, all keypoints are drawn.
    - links: A list of tuples, where each tuple contains two indices representing keypoints to connect.
    """
    
    # Make a copy of the image to draw on
    img_copy = image.copy()

    # Loop over each bounding box and corresponding keypoints

    if bbox is not None:
        x1, y1, x2, y2 = bbox

        # Draw the bounding box
        cv2.rectangle(img_copy, (int(x1), int(y1)), (int(x2), int(y2)), color=(0, 0, 255), thickness=2)  # Red color

    # Initialize a list to store the coordinates of the keypoints to be drawn

    if keypoints is not None:
        drawn_keypoints = []

        # Plot keypoints inside the bounding box
        for idx, keypoint in enumerate(keypoints):
            if filter_ids is None or (idx in filter_ids):
                offset_x, offset_y = keypoint
                keypoint_pos = (int(offset_x), int(offset_y))
                drawn_keypoints.append((idx, keypoint_pos))
                
                # Draw the keypoint
                cv2.circle(img_copy, keypoint_pos, radius=4, color=(0, 0, 255), thickness=-1)  # Blue color

    # Draw links between keypoints
    if links is not None:
        for link in links:
            id1, id2 = link
            if filter_ids is not None and (id1 not in filter_ids or (id2 not in filter_ids)): continue
            # Find the positions of the linked keypoints
            pos1 = (int(keypoints[id1][0]), int(keypoints[id1][1]))
            pos2 = (int(keypoints[id2][0]), int(keypoints[id2][1]))
            
            cv2.line(img_copy, pos1, pos2, color=(255, 0, 0), thickness=2)  # Green line for links

    return img_copy


def get_contour_from_mask(mask):
    # Ensure the mask is in the correct format (0 and 255)
    mask = (mask * 255).astype(np.uint8)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Select the largest contour (assuming the mask has a single object)
    if len(contours) > 0:
        # largest_contour = max(contours, key=cv2.contourArea)
        # all_contours = np.vstack(contours)

        return contours
    else:
        return None
    

def draw_contour_from_mask(image, mask, color=(0, 0, 255), thickness=2):
    
    contours = get_contour_from_mask(mask)
    # human_contour = human_contour.reshape(human_contour.shape[0], -1)
    if contours is not None:
        for contour in contours:
            cv2.drawContours(image, [contour], -1, color, thickness)
    return image


def compute_iou(bbox1, bbox2):
    """
    Computes the Intersection over Union (IoU) between two bounding boxes.
    
    Parameters:
    - bbox1: A list or tuple with four values (x_min, y_min, x_max, y_max) for the first bounding box
    - bbox2: A list or tuple with four values (x_min, y_min, x_max, y_max) for the second bounding box
    
    Returns:
    - iou: Intersection over Union (IoU) value between bbox1 and bbox2
    """
    # Unpack the coordinates of the bounding boxes
    x_min1, y_min1, x_max1, y_max1 = bbox1
    x_min2, y_min2, x_max2, y_max2 = bbox2

    # Determine the coordinates of the intersection rectangle
    x_min_inter = max(x_min1, x_min2)
    y_min_inter = max(y_min1, y_min2)
    x_max_inter = min(x_max1, x_max2)
    y_max_inter = min(y_max1, y_max2)

    # Compute the area of the intersection rectangle
    inter_width = max(0, x_max_inter - x_min_inter)
    inter_height = max(0, y_max_inter - y_min_inter)
    inter_area = inter_width * inter_height

    # Compute the area of both bounding boxes
    bbox1_area = (x_max1 - x_min1) * (y_max1 - y_min1)
    bbox2_area = (x_max2 - x_min2) * (y_max2 - y_min2)

    # Compute the area of the union
    union_area = bbox1_area + bbox2_area - inter_area

    # Compute the IoU
    iou = inter_area / union_area if union_area != 0 else 0.0

    return iou

def compute_iou_matrix(bboxes1, bboxes2):
    """
    Computes the Intersection over Union (IoU) between two sets of bounding boxes.
    
    Parameters:
    - bboxes1: (N, 4) numpy array where each row is (x_min, y_min, x_max, y_max)
    - bboxes2: (M, 4) numpy array where each row is (x_min, y_min, x_max, y_max)
    
    Returns:
    - iou_matrix: (N, M) numpy array where each element is the IoU between a bbox from bboxes1 and bboxes2
    """
    # Get number of boxes in bboxes1 (N) and bboxes2 (M)
    N = bboxes1.shape[0]
    M = bboxes2.shape[0]

    # Expand dimensions to enable broadcasting: (N, 1, 4) and (1, M, 4)
    bboxes1 = np.expand_dims(bboxes1, 1)  # Shape becomes (N, 1, 4)
    bboxes2 = np.expand_dims(bboxes2, 0)  # Shape becomes (1, M, 4)

    # Compute the coordinates of the intersection box
    x_min_inter = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    y_min_inter = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    x_max_inter = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    y_max_inter = np.minimum(bboxes1[..., 3], bboxes2[..., 3])

    # Compute the width and height of the intersection box
    inter_width = np.maximum(0, x_max_inter - x_min_inter)
    inter_height = np.maximum(0, y_max_inter - y_min_inter)

    # Compute the area of the intersection
    inter_area = inter_width * inter_height

    # Compute the area of both sets of boxes
    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])  # (N, 1)
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])  # (1, M)

    # Compute the union area
    union_area = area1 + area2 - inter_area

    # Compute IoU as the intersection over union
    iou_matrix = inter_area / np.maximum(union_area, 1e-6)  # Avoid division by zero

    return iou_matrix

def rgbd2pcd(rgb_image, depth_image, cam_K, mesh_xy=None):
    pass

def depth2pcd(depth_image, cam_K, mesh_xy=None):
    fx, fy, cx, cy = cam_K[0, 0], cam_K[1, 1], cam_K[0, 2], cam_K[1, 2]

    # Check if input is a NumPy array or PyTorch tensor
    is_numpy = isinstance(depth_image, np.ndarray)

    
    # Convert depth_image, x, and y to tensors if they are in NumPy format
    if is_numpy:
        H, W = depth_image.shape if mesh_xy is None else mesh_xy[0].shape
        depth_image = torch.from_numpy(depth_image).cuda()
    else:
        H, W = depth_image.shape if mesh_xy is None else mesh_xy[0].shape

    # Generate mesh grid (x, y) if mesh_xy is not provided
    if mesh_xy is None:
        x, y = torch.meshgrid(torch.arange(W, device=depth_image.device), torch.arange(H, device=depth_image.device), indexing='xy')
    else:
        # x, y = (torch.from_numpy(mesh_xy[0]).to(depth_image.device), 
                # torch.from_numpy(mesh_xy[1]).to(depth_image.device))
        x, y = mesh_xy

    
    # Calculate 3D coordinates
    x = (x - cx) * depth_image / fx
    y = (y - cy) * depth_image / fy
    z = depth_image

    # Stack coordinates to create a point cloud
    points = torch.stack((x, y, z), dim=-1).reshape(-1, 3)

    # Filter out points with zero depth
    # mask = points[:, 2] > 0
    # points = points[mask]

    # Return result in the same format as input
    return points.cpu().numpy() if is_numpy else points

def visualize_3d_point_cloud(rgb_image, depth_image, mask, cam_K):
    """
    Visualizes a 3D point cloud using RGB, depth, and instance mask images.
    
    Parameters:
    - rgb_image: (H, W, 3) RGB image
    - depth_image: (H, W) depth image with depth values in meters
    - mask: (N, H, W) binary instance masks
    - fx, fy, cx, cy: Intrinsic camera parameters for depth to 3D conversion
    """
    
    points, colors = rgbd2pcd(rgb_image, depth_image, cam_K)

    # print(points.shape)
    
    
    # Initialize Open3D point cloud
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    
    # Random colors for each instance
    np.random.seed(42)
    instance_colors = np.random.rand(mask.shape[0], 3)
    
    # Assign colors based on mask
    for i in range(mask.shape[0]):
        instance_mask = mask[i].flatten() > 0  # Get points for the i-th instance
        colors[instance_mask] = instance_colors[i]  # Set color for the instance

    # colors = colors[mask.cpu().numpy() > 0]
    # Update point cloud colors
    point_cloud.colors = o3d.utility.Vector3dVector(colors)

   
    # Visualize
    o3d.visualization.draw_geometries([point_cloud])
    # voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(point_cloud,
    #                                                         voxel_size=0.02)
    # o3d.visualization.draw_geometries([voxel_grid])

def gpu_voxel_filter(point_cloud, voxel_size, min_points_per_voxel):
    """
    GPU-based voxel filtering for a point cloud using PyTorch.
    
    Parameters:
    - point_cloud: (N, 3) torch.Tensor of point coordinates. ((N, 6) with colors)
    - voxel_size: float, the size of each voxel.
    - min_points_per_voxel: Minimum number of points required to keep points in a voxel.
    
    Returns:
    - filtered_points: torch.Tensor of filtered point coordinates.
    """
    if point_cloud.shape[-1] == 6:
        points, colors = point_cloud[:, :3], point_cloud[:, 3:]
    else:
        points, colors = point_cloud, None
        
    # Calculate voxel indices
    voxel_indices = torch.floor(points / voxel_size).int()

    # Use unique to get voxel occupancy counts
    unique_voxels, inverse_indices, counts = torch.unique(voxel_indices, return_inverse=True, return_counts=True, dim=0)

    # Mask for valid voxels based on point count
    valid_voxels_mask = counts >= min_points_per_voxel
    valid_voxel_indices = unique_voxels[valid_voxels_mask]

    # Filter points based on valid voxel occupancy
    mask = valid_voxels_mask[inverse_indices]
    filtered_points = points[mask]

    if colors is not None:
        filtered_colors = colors[mask]

        filtered_points = torch.cat([filtered_points, filtered_colors], -1)

    return filtered_points

def voxelize_point_cloud(point_cloud, voxel_size):
    """
    Voxelizes the point cloud by converting coordinates to voxel indices.
    """
    voxel_indices = torch.floor(point_cloud / voxel_size).int()
    return voxel_indices

def filter_voxels_by_occupancy(voxels, min_points_per_voxel):
    """
    Filters out voxels with occupancy below a specified threshold.
    """
    unique_voxels, inverse_indices, counts = torch.unique(voxels, return_inverse=True, return_counts=True, dim=0)
    valid_voxel_mask = counts >= min_points_per_voxel
    valid_voxel_indices = unique_voxels[valid_voxel_mask]

    voxel_indices = torch.arange(voxels.shape[0]).to(device=voxels.device)
    
    # Mask points based on valid voxels
    mask = valid_voxel_mask[inverse_indices]
    filtered_voxel_indices = voxel_indices[mask]
    
    return valid_voxel_indices, filtered_voxel_indices

def connected_components_on_voxel_grid(voxel_indices):
    """
    Performs connected component analysis on a voxel grid.
    """
    # Get grid dimensions and shift indices
    min_coords = voxel_indices.min(dim=0).values
    max_coords = voxel_indices.max(dim=0).values
    grid_shape = (max_coords - min_coords + 1).cpu().numpy()
    shifted_indices = (voxel_indices - min_coords).cpu().numpy()
    
    # Create binary voxel grid
    voxel_grid = np.zeros(grid_shape, dtype=bool)
    voxel_grid[shifted_indices[:, 0], shifted_indices[:, 1], shifted_indices[:, 2]] = True
    
    # Label connected components
    labeled_grid, num_components = label(voxel_grid)
    
    # Map components back to original voxel indices
    component_labels = labeled_grid[shifted_indices[:, 0], shifted_indices[:, 1], shifted_indices[:, 2]]
    
    return component_labels, num_components

def neighbor_search(init_points, points, thres=0.05):
    indices = torch.arange(points.shape[0]).to(points.device)


    connected = torch.zeros_like(indices).bool()

    query_points = init_points
    while len(indices) > 0:

        remain_points = points[indices]
        distance = torch.cdist(remain_points, query_points) 

        is_neighbor = (distance < thres).any(-1)

        if not is_neighbor.any(): break
        
        connected[indices[is_neighbor]] = True

        query_points = remain_points[is_neighbor]

        indices = indices[~is_neighbor]

    return points[connected], connected


def contact_estimation(rgb_image, 
                       depth_image, 
                       mask, cam_K, 
                       mesh_xy=None, 
                       voxel_size=0.01, 
                       min_points_per_voxel=3, 
                       contact_threshold=1, 
                       hand_kpts=None):
    with torch.profiler.profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        import time
        start = time.time()
        # human_depth = depth_image * mask
        # human_pnts, human_clrs = rgbd2pcd(rgb_image, human_depth, cam_K, mesh_xy)
        # print('here', mesh_xy[0].dtype)
        pnts = depth2pcd(torch.from_numpy(depth_image).cuda(), cam_K, mesh_xy).float()

        # other_depth = depth_image * (1 - mask)
        # other_pnts, other_clrs = rgbd2pcd(rgb_image, other_depth, cam_K, mesh_xy)
        mask_tensor = torch.from_numpy(mask).cuda().reshape(-1, 1)
        human_pnts = pnts * mask_tensor
        other_pnts = pnts * (1 - mask_tensor)

        human_mask = human_pnts[:, 2] > 0
        other_mask = other_pnts[:, 2] > 0
        human_pnts = human_pnts[human_mask]
        other_pnts = other_pnts[other_mask]

       
        # filter_human_pnts = gpu_voxel_filter(torch.from_numpy(human_pnts).cuda(), 0.02, 5)
        # start0 = time.time()

        # Voxelize both human and non-human point clouds
        human_voxels = voxelize_point_cloud(human_pnts, voxel_size)
        other_voxels = voxelize_point_cloud(other_pnts, voxel_size)

        # start1 = time.time()
        # print('start', human_voxels.shape, other_voxels.shape, start1 - start0)

        # Filter out low-occupancy voxels
        human_valid_voxels, human_voxel_indices = filter_voxels_by_occupancy(human_voxels, min_points_per_voxel)
        other_valid_voxels, other_voxel_indices = filter_voxels_by_occupancy(other_voxels, min_points_per_voxel)

        # start2 = time.time()
        # Connected components on voxel grids
        # human_labels, human_components = connected_components_on_voxel_grid(human_voxel_indices)
        # other_labels, other_components = connected_components_on_voxel_grid(other_voxel_indices)
        # print('start1', human_voxel_indices.shape, other_voxel_indices.shape, start2 - start1)
        # print('start2', human_valid_voxels.shape, other_valid_voxels.shape)

        human_valid_pnts = human_pnts[human_voxel_indices]
        other_valid_pnts = other_pnts[other_voxel_indices]

        

        if hand_kpts is not None:
            fx, fy, cx, cy = cam_K[0, 0], cam_K[1, 1], cam_K[0, 2], cam_K[1, 2]
            left_hand_2d, right_hand_2d = hand_kpts
            left_hand_z = depth_image[int(left_hand_2d[1]), int(left_hand_2d[0])]
            right_hand_z = depth_image[int(right_hand_2d[1]), int(right_hand_2d[0])]

            left_hand_3d = torch.Tensor([(left_hand_2d[0] - cx)/fx, (left_hand_2d[1] - cy)/fy, 1]) * left_hand_z
            right_hand_3d = torch.Tensor([(right_hand_2d[0] - cx)/fx, (right_hand_2d[1] - cy)/fy, 1]) * right_hand_z

            left_hand_3d = left_hand_3d.to(pnts.device)
            right_hand_3d = right_hand_3d.to(pnts.device)

            hand_3d = torch.stack([left_hand_3d, right_hand_3d], 0)

            hand_dist = torch.cdist(human_valid_pnts, hand_3d)
            left_dist = hand_dist[:, 0]
            right_dist = hand_dist[:, 1]

            left_valid_pnts = human_valid_pnts[left_dist < 0.1]
            right_valid_pnts = human_valid_pnts[right_dist < 0.1]

            left_hand_pnts, _ = neighbor_search(init_points=left_hand_3d[None],
                                                     points=left_valid_pnts)
            
            right_hand_pnts, _ = neighbor_search(init_points=right_hand_3d[None],
                                                     points=right_valid_pnts)


            hand_to_object_dist = torch.cdist(other_valid_pnts, hand_3d)

            other_valid_pnts = other_valid_pnts[(hand_to_object_dist < 1).any(-1)]

            _, left_connected_mask = neighbor_search(init_points=left_hand_3d[None],
                                                        points=other_valid_pnts)
            
            _, right_connected_mask = neighbor_search(init_points=right_hand_3d[None],
                                                        points=other_valid_pnts)
            
            if left_connected_mask.sum() < 100:
                left_connected_mask = torch.zeros_like(left_connected_mask)
        
            if right_connected_mask.sum() < 100:
                right_connected_mask = torch.zeros_like(right_connected_mask)
            
            other_valid_pnts = other_valid_pnts[left_connected_mask | right_connected_mask]

            left_hand_contact_dist = torch.cdist(left_hand_pnts, other_valid_pnts)
            right_hand_contact_dist = torch.cdist(right_hand_pnts, other_valid_pnts)

            left_hand_contact_num = (left_hand_contact_dist < 0.05).sum(-1)
            right_hand_contact_num = (right_hand_contact_dist < 0.05).sum(-1)

            contact_pnts = torch.cat([left_hand_pnts[left_hand_contact_num > 2], right_hand_pnts[right_hand_contact_num > 2]], 0) 

        else:
            human_cen = human_valid_pnts.mean(0)
            distance = torch.norm(other_valid_pnts - human_cen, dim=-1)

            other_valid_pnts = other_valid_pnts[distance < 3]

        # print(human_valid_pnts.shape, other_valid_pnts.shape)
        if other_valid_pnts.shape[0] > 0:

            # points = torch.cat([human_valid_pnts, other_valid_pnts], 0).cpu().numpy()
            dist = torch.cdist(other_valid_pnts, hand_3d)

            # contact_pnts = other_valid_pnts[(dist < 0.1).any(-1)].cpu().numpy()
            other_valid_pnts = other_valid_pnts.cpu().numpy()
            contact_pnts = contact_pnts.cpu().numpy()

            
            
        else:
            points = human_valid_pnts.cpu().numpy()
            other_valid_pnts = None
            contact_pnts = None
        # points = human_valid_pnts.cpu().numpy()
        human_valid_pnts = human_valid_pnts.cpu().numpy()
      

        # Convert voxel indices to sets for efficient adjacency check
    return human_valid_pnts, other_valid_pnts, contact_pnts

    # print(prof.key_averages().table(sort_by="cuda_time_total" if torch.cuda.is_available() else "cpu_time_total"))

filter_ids = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22] #,91,112]
index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(filter_ids)}


def filter_keypoints_and_links(keypoints, links):
    """
    Filters the keypoints and links based on filter_ids.

    Parameters:
    keypoints (np.ndarray): A (N, 3) array representing the keypoints.
    links (list of tuples): A list of (idx1, idx2) representing links between keypoints.
    filter_ids (list): A list of keypoint indices to keep.

    Returns:
    filtered_keypoints (np.ndarray): A (len(filter_ids), 3) array of filtered keypoints.
    filtered_links (list of tuples): A list of updated (idx1, idx2) based on filtered keypoints.
    """
    # Step 1: Filter the keypoints based on filter_ids
    filtered_keypoints = keypoints[filter_ids]

    # Step 2: Create a mapping from old indices to new indices
    

    # Step 3: Update the links to the new indices and filter out invalid ones
    filtered_links = []
    for idx1, idx2 in links:
        if (idx1 in index_map) and (idx2 in index_map):
            filtered_links.append((index_map[idx1], index_map[idx2]))

    return filtered_keypoints, filtered_links