import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from realsense2_camera_msgs.msg import Extrinsics
import numpy as np
import cv2
from cv_bridge import CvBridge
from ultralytics import YOLO
import tf2_ros
from geometry_msgs.msg import TransformStamped
from alphabeta import Tic, get_enemy, determine

class SHOWCAM(Node):
    def __init__(self):
        super().__init__('play_tictactoe')
        self.bridge = CvBridge()
        self.camera_subscription = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.camera_callback,
            10)
        self.depth_subscription = self.create_subscription(
            Image,
            '/camera/camera/aligned_depth_to_color/image_raw',
            self.depth_callback,
            10)
        self.extrinsics_subscription = self.create_subscription(
            Extrinsics,
            '/camera/camera/extrinsics/depth_to_color',
            self.extrinsics_callback,
            10)
        self.camera_info_subscription = self.create_subscription(
            CameraInfo,
            '/camera/camera/aligned_depth_to_color/camera_info',
            self.camera_info_callback,
            10)
        self.camera_subscription  # prevent unused variable warning
        self.depth_subscription  # prevent unused variable warning
        self.extrinsics_subscription  # prevent unused variable warning
        self.camera_info_subscription  # prevent unused variable warning
        self.extrinsics_rotation = None
        self.extrinsics_translation = None
        self.intrinsics = None
        self.depth_image = None
        self.color_image = None
        self.mask = None
        self.timer = self.create_timer(0.1, self.detect_and_print_coordinates_debug)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.timer2 = self.create_timer(0.1, self.get_transform)
        # Load YOLO model
        model_path = "/home/stu/tic-tac-toe_test_yolo_20.2.2025/tic-tac-toe_test_yolo/datasets_tic_tac_toe_v1/best_tictactoe_v1.pt"
        self.model = YOLO(model_path)
        self.class_names = ['None', 'o', 'table', 'x']
        self.target_class = 'table'
        self.T = None
        self.save_grid = None
        self.save_symbol_pos = None
        self.get_logger().info('Node initialized')
        

    def get_transform(self):
        from_frame = 'head_camera'
        to_frame = 'gen3lite_base_link'
        try:
            trans: TransformStamped = self.tf_buffer.lookup_transform(to_frame, from_frame, rclpy.time.Time())

            tx, ty, tz = trans.transform.translation.x, trans.transform.translation.y, trans.transform.translation.z
            qx, qy, qz, qw = trans.transform.rotation.x, trans.transform.rotation.y, trans.transform.rotation.z, trans.transform.rotation.w

            self.T = self.to_transformation_matrix(tx, ty, tz, qx, qy, qz, qw)
            #self.get_logger().info(f'Transformation Matrix from {from_frame} to {to_frame}:\n{self.T}')
            #transform camera coordinates to robot base coordinates
            
            # camera_coordinates = [0, 0, 0.5, 1]
            # robot_coordinates = np.dot(T, camera_coordinates)
            # self.get_logger().info(f'Camera coordinates: {camera_coordinates}')
            # self.get_logger().info(f'Robot coordinates: {robot_coordinates}')

        except Exception as e:
            self.get_logger().warn(f'Could not get transform: {str(e)}')
            #pass

    def to_transformation_matrix(self, tx, ty, tz, qx, qy, qz, qw):
        from scipy.spatial.transform import Rotation
        rot = Rotation.from_quat([qx, qy, qz, qw])
        T = np.eye(4)
        T[:3, :3] = rot.as_matrix()
        T[:3, 3] = [tx, ty, tz]
        return T



    def camera_callback(self, msg):
        #self.get_logger().info('Received camera image data')
        self.color_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def depth_callback(self, msg):
        #self.get_logger().info('Received depth image data')
        self.depth_image = msg

    def extrinsics_callback(self, msg):
        #self.get_logger().info('Received extrinsics data')
        self.extrinsics_rotation = np.array(msg.rotation).reshape(3, 3)
        self.extrinsics_translation = np.array(msg.translation)

    def camera_info_callback(self, msg):
        #self.get_logger().info('Received camera info data')
        self.intrinsics = msg

    def calculate_real_world_coordinates(self, u, v):
        if self.intrinsics is None:
            self.get_logger().warn('Intrinsics not received yet')
            return None

        
            # Convert depth image to numpy array
        depth_array = np.frombuffer(self.depth_image.data, dtype=np.uint16).reshape(self.depth_image.height, self.depth_image.width)

        # Get depth value at (u, v)
        depth = depth_array[v, u]
        # Get camera intrinsics
        fx = self.intrinsics.k[0]
        fy = self.intrinsics.k[4]
        cx = self.intrinsics.k[2]
        cy = self.intrinsics.k[5]

        # Calculate real-world coordinates
        z = depth / 1000.0  # Convert depth from mm to meters if necessary
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy

        # Apply extrinsics transformation if available
        if self.extrinsics_rotation is not None and self.extrinsics_translation is not None:
            point = np.array([x, y, z])
            transformed_point = self.extrinsics_rotation @ point + self.extrinsics_translation
            return transformed_point

        return np.array([x, y, z])
    
    def calculate_robot_coordinates(self, camera_coordinates):
        if self.T is None:
            self.get_logger().warn('Transformation matrix not received yet')
            return None
        
        x, y, z = camera_coordinates
        camera_coordinates = np.array([x, y, z, 1])
        
        robot_coordinates = np.dot(self.T, camera_coordinates)
        self.get_logger().info(f'Robot coordinates: {robot_coordinates}')
        return robot_coordinates
        # Convert depth image to numpy array
        
    def robot_draw_at_position(self, symbol, position):
        # Convert position to camera coordinates
        x, y, z = position
        camera_coordinates = np.array([x, y, z, 1])
        robot_coordinates = self.calculate_robot_coordinates(camera_coordinates)
        self.get_logger().info(f'Robot coordinates: {robot_coordinates}')
        # Draw symbol at robot coordinates
        #self.draw_symbol_at(symbol, robot_coordinates)
        
    def draw_symbol_at(self, symbol, coordinates):
        # Draw symbol on the image
        center_x, center_y, center_z = coordinates
        if symbol == 'x':
            # draw X at center_x, center_y
            pass
        elif symbol == 'o':
            # draw o at center_x, center_y
            pass
        else:
            return
    
        
        

    def detect_and_print_coordinates_debug(self):
        if self.color_image is None or self.depth_image is None:
            self.get_logger().warn('Color or depth image not received yet')
            return

        # Perform object detection
        results = self.model(source=self.color_image, conf=0.8, verbose=False)
        #self.get_logger().info('Start detecting board')

        

        for result in results:
                    boxes = result.boxes
                    if boxes is None:
                        #print("No boxes found in result.boxes, checking obb object")
                        
                        obb = result.obb
                        if obb is not None:
                            
                            class_ids = obb.cls.int().tolist()
                            confidences = obb.conf.tolist()
                            coordinates = obb.xyxy.int().tolist()
                        else:
                            continue
                    else:
                        class_ids = boxes.cls.int().tolist()
                        confidences = boxes.conf.tolist()
                        coordinates = boxes.xyxy.int().tolist()
                    for class_id, confidence, (x1, y1, x2, y2) in zip(class_ids, confidences, coordinates):
                        class_name = self.class_names[class_id]
                        #print(f"Class: {class_name}, Confidence: {confidence}, Coordinates: {x1, y1, x2, y2}")
                        if class_name == self.target_class:
                            self.get_logger().info('Found target class')
                            x1-= 20
                            y1-= 20
                            x2+= 20
                            y2+= 20
                            roi = self.color_image[y1:y2, x1:x2]
                            self.mask = np.zeros_like(self.color_image)
                            self.mask[y1:y2, x1:x2] = roi
                            
                            roi_results = self.model(source=self.mask, conf=0.75, verbose=False, show=False)
                        
                            for roi_result in roi_results:
                                roi_boxes = roi_result.boxes
                                if roi_boxes is None:
                                    roi_obb = roi_result.obb
                                    if roi_obb is not None:
                                        # remove class named "table" from roi_obb
                                        roi_class_ids = roi_obb.cls.int().tolist()
                                        roi_confidences = roi_obb.conf.tolist()
                                        roi_coordinates = roi_obb.xyxy.int().tolist()
                                        roi_center = roi_obb.xywhr.int().tolist()
                                    else:
                                        continue
                                else:    
                                    # remove class named "table" from roi_boxes
                                    roi_boxes = [box for box in roi_boxes if self.class_names[box.cls.int().item()] != 'table']
                                    roi_class_ids = roi_boxes.cls.int().tolist()
                                    roi_confidences = roi_boxes.conf.tolist()
                                    roi_coordinates = roi_boxes.xyxy.int().tolist()
                                    roi_center = roi_boxes.xywhr.int().tolist()
                                    
                                    
                                    
                                grid = [[[] for _ in range(3)] for _ in range(3)]   
                                detected_symbols = []
                                for roi_class_id, roi_confidence, (rx1, ry1, rx2, ry2), (rcx,rcy,rw,rh,rr) in zip(roi_class_ids, roi_confidences, roi_coordinates, roi_center):
                                                                    
                                    symbol_name = self.class_names[roi_class_id]
                                    if symbol_name == 'table':
                                        continue
                                    
                                    # center_x = (rx1 + rx2) / 2
                                    # center_y = (ry1 + ry2) / 2
                                    center_x = rcx
                                    center_y = rcy
                                    
                                    real_world_x, real_world_y, real_world_z = self.calculate_real_world_coordinates(center_x, center_y)
                                    
                                    detected_symbols.append({
                                        'class': symbol_name,
                                        'confidence': roi_confidence,
                                        'coordinates': (rx1, ry1, rx2, ry2),
                                        'center': (center_x, center_y),
                                        'real_world_position': (real_world_x, real_world_y, real_world_z)
                                    })
                                
                                for symbol in detected_symbols:
                                    center_x, center_y = symbol['center']
                                    real_world_position = tuple(round(coord, 2) for coord in symbol['real_world_position'])
                                    label = str(real_world_position)
                                    
                                    cv2.circle(self.mask, (center_x, center_y), 5, (0, 255, 0), -1)
                                    #also add the label of class
                                    #cv2.putText(mask, label, (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                                    
                                #cv2.imshow("debug",mask)
                                
                                # check if center_y is in the range of +- 20 pixels then put them in the same row
                                detected_symbols.sort(key=lambda symbol: symbol['center'][1])
                                #print("sort on center_y")
                               
                                
                                #sort center_x in each row individually
                                for i in range(0, len(detected_symbols), 3):
                                    detected_symbols[i:i+3] = sorted(detected_symbols[i:i+3], key=lambda symbol: symbol['center'][0])
                                    
                                # [For Debug] Show the image with only position number 5 after sorted
                                if len(detected_symbols) > 4:
                                    center_x, center_y = detected_symbols[4]['center']
                                    real_world_position = detected_symbols[4]['real_world_position']
                                    cv2.circle(self.mask, (center_x, center_y), 10, (0, 0, 255), -1)
                                    robot_coordinates = self.calculate_robot_coordinates(real_world_position)
                                    label = f"({robot_coordinates[0]:.2f}, {robot_coordinates[1]:.2f}, {robot_coordinates[2]:.2f})"
                                    cv2.putText(self.mask, label, (center_x, center_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                                    cv2.imshow("Position 5", self.mask)
                                    self.get_logger().info(f"Position 5 real-world coordinates: {real_world_position}")
                                    cv2.waitKey(1)
                                    
                                # Extract class names
                                board_status = [symbol['class'] for symbol in detected_symbols]
                                symbol_localtion = [symbol['center'] for symbol in detected_symbols]

                                # Rearrange class names into a 3x3 grid
                                grid = [board_status[i:i+3] for i in range(0, len(board_status), 3)]
                                symbol_pos = [symbol_localtion[i:i+3] for i in range(0, len(symbol_localtion), 3)]
                                

                                
                                #check if the grid has 3x3 size
                                if len(grid) == 3 and all(len(row) == 3 for row in grid):
                                        if self.save_grid != grid:
                                            self.get_logger().info("New grid detected")
                                            
                                            self.save_grid = grid
                                            self.save_symbol_pos = symbol_pos
                                            return self.save_grid, self.save_symbol_pos
                                
                                
                        
                        else:
                            self.get_logger().info('target class not found')

    def update_game_board(self):
        if self.color_image is None or self.depth_image is None:
            self.get_logger().warn('Color or depth image not received yet')
            return

        # Perform object detection
        results = self.model(source=self.color_image, conf=0.8, verbose=True)

        

        for result in results:
                    boxes = result.boxes
                    if boxes is None:
                        #print("No boxes found in result.boxes, checking obb object")
                        
                        obb = result.obb
                        if obb is not None:
                            
                            class_ids = obb.cls.int().tolist()
                            confidences = obb.conf.tolist()
                            coordinates = obb.xyxy.int().tolist()
                        else:
                            continue
                    else:
                        class_ids = boxes.cls.int().tolist()
                        confidences = boxes.conf.tolist()
                        coordinates = boxes.xyxy.int().tolist()
                    for class_id, confidence, (x1, y1, x2, y2) in zip(class_ids, confidences, coordinates):
                        class_name = self.class_names[class_id]
                        #print(f"Class: {class_name}, Confidence: {confidence}, Coordinates: {x1, y1, x2, y2}")
                        if class_name == self.target_class:
                            self.get_logger().info('Found target class')
                            x1-= 20
                            y1-= 20
                            x2+= 20
                            y2+= 20
                            roi = self.color_image[y1:y2, x1:x2]
                            self.mask = np.zeros_like(self.color_image)
                            self.mask[y1:y2, x1:x2] = roi
                            
                            roi_results = self.model(source=self.mask, conf=0.75, verbose=False, show=False)
                        
                            for roi_result in roi_results:
                                roi_boxes = roi_result.boxes
                                if roi_boxes is None:
                                    roi_obb = roi_result.obb
                                    if roi_obb is not None:
                                        # remove class named "table" from roi_obb
                                        roi_class_ids = roi_obb.cls.int().tolist()
                                        roi_confidences = roi_obb.conf.tolist()
                                        roi_coordinates = roi_obb.xyxy.int().tolist()
                                        roi_center = roi_obb.xywhr.int().tolist()
                                    else:
                                        continue
                                else:    
                                    # remove class named "table" from roi_boxes
                                    roi_boxes = [box for box in roi_boxes if self.class_names[box.cls.int().item()] != 'table']
                                    roi_class_ids = roi_boxes.cls.int().tolist()
                                    roi_confidences = roi_boxes.conf.tolist()
                                    roi_coordinates = roi_boxes.xyxy.int().tolist()
                                    roi_center = roi_boxes.xywhr.int().tolist()
                                    
                                    
                                    
                                grid = [[[] for _ in range(3)] for _ in range(3)]   
                                detected_symbols = []
                                for roi_class_id, roi_confidence, (rx1, ry1, rx2, ry2), (rcx,rcy,rw,rh,rr) in zip(roi_class_ids, roi_confidences, roi_coordinates, roi_center):
                                                                    
                                    symbol_name = self.class_names[roi_class_id]
                                    if symbol_name == 'table':
                                        continue
                                    
                                    # center_x = (rx1 + rx2) / 2
                                    # center_y = (ry1 + ry2) / 2
                                    center_x = rcx
                                    center_y = rcy
                                    
                                    real_world_x, real_world_y, real_world_z = self.calculate_real_world_coordinates(center_x, center_y)
                                    
                                    detected_symbols.append({
                                        'class': symbol_name,
                                        'confidence': roi_confidence,
                                        'coordinates': (rx1, ry1, rx2, ry2),
                                        'center': (center_x, center_y),
                                        'real_world_position': (real_world_x, real_world_y, real_world_z)
                                    })
                                
                                for symbol in detected_symbols:
                                    center_x, center_y = symbol['center']
                                    real_world_position = tuple(round(coord, 2) for coord in symbol['real_world_position'])
                                    label = str(real_world_position)
                                    
                                    cv2.circle(self.mask, (center_x, center_y), 5, (0, 255, 0), -1)
                                    #also add the label of class
                                    #cv2.putText(mask, label, (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                                    
                                #cv2.imshow("debug",mask)
                                
                                # check if center_y is in the range of +- 20 pixels then put them in the same row
                                detected_symbols.sort(key=lambda symbol: symbol['center'][1])
                                #print("sort on center_y")
                               
                                
                                #sort center_x in each row individually
                                for i in range(0, len(detected_symbols), 3):
                                    detected_symbols[i:i+3] = sorted(detected_symbols[i:i+3], key=lambda symbol: symbol['center'][0])
                                    
                                # [For Debug] Show the image with only position number 5 after sorted
                                # if len(detected_symbols) > 4:
                                #     center_x, center_y = detected_symbols[4]['center']
                                #     real_world_position = detected_symbols[4]['real_world_position']
                                #     cv2.circle(self.mask, (center_x, center_y), 10, (0, 0, 255), -1)
                                #     label = f"({real_world_position[0]:.2f}, {real_world_position[1]:.2f}, {real_world_position[2]:.2f})"
                                #     cv2.putText(self.mask, label, (center_x, center_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                                #     cv2.imshow("Position 5", self.mask)
                                #     self.get_logger().info(f"Position 5 real-world coordinates: {real_world_position}")
                                #     cv2.waitKey(1)
                                    
                                # Extract class names
                                board_status = [symbol['class'] for symbol in detected_symbols]
                                symbol_localtion = [symbol['center'] for symbol in detected_symbols]

                                # Rearrange class names into a 3x3 grid
                                grid = [board_status[i:i+3] for i in range(0, len(board_status), 3)]
                                symbol_pos = [symbol_localtion[i:i+3] for i in range(0, len(symbol_localtion), 3)]
                                

                                
                                #check if the grid has 3x3 size
                                if len(grid) == 3 and all(len(row) == 3 for row in grid):
                                        if self.save_grid != grid:
                                            print("New grid detected")
                                            
                                            self.save_grid = grid
                                            self.save_symbol_pos = symbol_pos
                                            return self.save_grid, self.save_symbol_pos
                                
                                
                        
                        else:
                            self.get_logger().info('target class not found')


def play_game(detector):
    board = Tic()
    board.show()
    
    # Check if the board from camera is empty or not
    grid, pos = detector.update_game_board()
    if all(cell == 'None' for row in grid for cell in row):
        print("Detected board is empty")
    else:
        print("Detected board is not empty")
        return "Please clean the board before starting the game"
    
    
    
    if board.squares == [None, None, None, None, None, None, None, None, None]:
        print("Board is empty")
    else:
        print("Board is not empty")
    

    while not board.complete():
        player = 'x'
        #show_player_move_popup()
        print('Processing frames to determine player move...')
        grid, pos = detector.update_game_board()  # Get the 3x3 string grid
        player_move = convert_grid_to_move(grid, board.squares)
        if player_move not in board.available_moves():
            print('Invalid move detected. Please try again.')
            continue
        board.make_move(player_move, player)
        print("Player move: ", player_move)
        board.show()
        cv2.imshow('Tic Tac Toe', draw_board(board))
        cv2.waitKey(500)

        if board.complete():
            break
        player = get_enemy(player)
        #show_computer_move_popup()
        computer_move = determine(board, player)
        
        #send the computer move to the robot
        send_move_to_robot(computer_move, pos)
        
        
        board.make_move(computer_move, player)
        print("Computer move: ", computer_move)
        board.show()
        cv2.imshow('Tic Tac Toe', draw_board(board))
        cv2.waitKey(500)
    print('Winner is', board.winner())
    
    cv2.waitKey(2000)
    cv2.destroyAllWindows()
    
def draw_board(board):
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    for i in range(1, 3):
        cv2.line(img, (0, i * 100), (300, i * 100), (255, 255, 255), 2)
        cv2.line(img, (i * 100, 0), (i * 100, 300), (255, 255, 255), 2)
    for i, square in enumerate(board.squares):
        if square is not None:
            x, y = (i % 3) * 100 + 50, (i // 3) * 100 + 50
            if square == 'x':
                cv2.line(img, (x - 25, y - 25), (x + 25, y + 25), (255, 0, 0), 2)
                cv2.line(img, (x + 25, y - 25), (x - 25, y + 25), (255, 0, 0), 2)
            elif square == 'o':
                cv2.circle(img, (x, y), 25, (0, 255, 0), 2)
    return img
    
def send_move_to_robot(computer_move, pos):
    # Get the center of the square
    x, y, z = pos[computer_move // 3][computer_move % 3]
    #convert the camera coordinate to robot coordinate
    
    print(f"Sending move to robot: {x}, {y}, {z}")
    
    # Send the move to the robot
def convert_grid_to_move(grid, current_squares):
    """
    Convert the 3x3 string grid to a move index.
    The grid is a list of lists with 'X', 'O', or None.
    """
    for i in range(3):
        for j in range(3):
            index = i * 3 + j
            if grid[i][j] == 'x' and current_squares[index] is None:
                return index
    return -1  # Return an invalid move if no valid move is found
            


def main():
    rclpy.init()
    node = SHOWCAM()
    
    #play_game(node)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()