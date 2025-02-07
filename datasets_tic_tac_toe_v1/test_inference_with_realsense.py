from ultralytics import YOLO
import cv2
import pyrealsense2 as rs
import numpy as np
import threading
import logging
from pynput import keyboard
import tkinter as tk
from tkinter import messagebox
from alphabeta import Tic, get_enemy, determine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model_path = "/home/vitvasin/ultralytics/tic-tac-toe_test_yolo/datasets_tic_tac_toe_v1/best_tictactoe_v1.pt"
class_names = ['None', 'o', 'table', 'x']
#symbol_names = ['null', 'o', 'x']
target_class = 'table'
save_grid=[]
save_symbol_pos=[]
class ElevatorPanelDetector:
    TIMEOUT_DURATION = 20
    offset_x = 40
    offset_y = -5
    offset_z = 140
    pressed = False

    def __init__(self, model_path, class_names, target_class):
        self.model = YOLO(model_path)
        self.class_names = class_names
        self.target_class = target_class
        self.save_grid = save_grid
        self.save_symbol_pos = save_symbol_pos
        #self.symbol_names = symbol_names
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
        self.pipeline.start(self.config)
        self.depth_scale = self.pipeline.get_active_profile().get_device().first_depth_sensor().get_depth_scale()
        self.stop_event = threading.Event()

    def process_frames(self):
        try:
            while not self.stop_event.is_set():
                frames = self.pipeline.wait_for_frames()
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()
                if not depth_frame or not color_frame:
                    continue

                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())

                results = self.model(source=color_image, conf=0.8, verbose=False, show=False)
                
               # print(results)
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
                            x1-= 20
                            y1-= 20
                            x2+= 20
                            y2+= 20
                            roi = color_image[y1:y2, x1:x2]
                            mask = np.zeros_like(color_image)
                            mask[y1:y2, x1:x2] = roi
                            
                            roi_results = self.model(source=mask, conf=0.75, verbose=False, show=False)
                        
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
                                
                                #grid = [['' for _ in range(3)] for _ in range(3)]
                                
                                # Define the 3x3 grid with lists
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
                                    
                                    
                                    detected_symbols.append({
                                        'class': symbol_name,
                                        'confidence': roi_confidence,
                                        'coordinates': (rx1, ry1, rx2, ry2),
                                        'center': (center_x, center_y)
                                    })
                                    
                                #debug by dot the center_x, center_y on mask
                                for symbol in detected_symbols:
                                    center_x, center_y = symbol['center']
                                    label = symbol['class'] + '\n' +str(symbol['center'])
                                    
                                    cv2.circle(mask, (center_x, center_y), 5, (0, 255, 0), -1)
                                    #also add the label of class
                                    cv2.putText(mask, label, (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
                                    
                                cv2.imshow("debug",mask)
                                
                                # check if center_y is in the range of +- 20 pixels then put them in the same row
                                detected_symbols.sort(key=lambda symbol: symbol['center'][1])
                                #print("sort on center_y")
                               # for symbol in detected_symbols:
                                #    print(symbol['class'] + " at " + str(symbol['center']))
                                
                                #sort center_x in each row individually
                                for i in range(0, len(detected_symbols), 3):
                                    detected_symbols[i:i+3] = sorted(detected_symbols[i:i+3], key=lambda symbol: symbol['center'][0])
                                
                               # print("Reaarange on center_x for each row")
                                #for symbol in detected_symbols:
                                #    print(symbol['class'] + " at " + str(symbol['center']))
                                
                                
                                
                                
                                
                                
                                # print("UnSorted class:")
                                # for symbol in detected_symbols:
                                #     print(symbol['class'] + " at " + str(symbol['center']))
                                
                                #tolerance = 1e-5
                                #detected_symbols.sort(key=lambda symbol: (round(symbol['center'][1] / tolerance) * tolerance, symbol['center'][0]))

                                #detected_symbols.sort(key=lambda symbol: (symbol['center'][1], symbol['center'][0]))
                                #detected_symbols = sorted(detected_symbols, key=itemgetter('center'))
                                


                                # Extract class names
                                board_status = [symbol['class'] for symbol in detected_symbols]
                                symbol_localtion = [symbol['center'] for symbol in detected_symbols]

                                # Rearrange class names into a 3x3 grid
                                grid = [board_status[i:i+3] for i in range(0, len(board_status), 3)]
                                symbol_pos = [symbol_localtion[i:i+3] for i in range(0, len(symbol_localtion), 3)]
                                

                                
                                #check if the grid has 3x3 size
                                if len(grid) == 3 and all(len(row) == 3 for row in grid):
#                                     # Get the location in camera coordinate from depth image
                                    symbol_pos = [
                                        [
                                            (x, y, depth_image[int(y)][int(x)] * self.depth_scale)
                                            for x, y in row
                                        ]
                                        for row in symbol_pos
                                    ]

                                    # Convert symbol positions to point cloud positions
                                    symbol_pos = [
                                        [
                                            rs.rs2_deproject_pixel_to_point(
                                                self.pipeline.get_active_profile().get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics(),
                                                [x, y],
                                                z
                                            )
                                            for x, y, z in row
                                        ]
                                        for row in symbol_pos
                                     ]
                                    
                                    if self.save_grid != grid:
                                        self.save_grid = grid
                                        self.save_symbol_pos = symbol_pos
                                        print("New grid detected")
                                        # for row in grid:
                                        #     print(row)
                                        
                                        return self.save_grid, self.save_symbol_pos
                                    
                                    

                
        finally:
        #    self.pipeline.stop()
            print("finished!")
        #    cv2.destroyAllWindows()

wait_for_player_move = True
# def play_game(detector):
#     board = Tic()
#     board.show()

#     while not board.complete():
#         player = 'x'
#         print('Processing frames to determine player move...')
#         grid, pos = detector.process_frames()  # Get the 3x3 string grid
#         player_move = convert_grid_to_move(grid, board.squares)
#         if player_move not in board.available_moves():
#             print('Invalid move detected. Please try again.')
#             continue
#         board.make_move(player_move, player)
#         board.show()

#         if board.complete():
#             break
#         player = get_enemy(player)
#         computer_move = determine(board, player)
#         board.make_move(computer_move, player)
#         board.show()
#     print('Winner is', board.winner())
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

def play_game(detector):
    board = Tic()
    board.show()
    
    # Check if the board from camera is empty or not
    grid, pos = detector.process_frames()
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
        grid, pos = detector.process_frames()  # Get the 3x3 string grid
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
        board.make_move(computer_move, player)
        print("Computer move: ", computer_move)
        board.show()
        cv2.imshow('Tic Tac Toe', draw_board(board))
        cv2.waitKey(500)
    print('Winner is', board.winner())
    show_winner_popup(board.winner())
    cv2.waitKey(2000)
    cv2.destroyAllWindows()

def show_winner_popup(winner):
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    messagebox.showinfo("Game Result", f"Winner is {winner}")
    root.destroy()
    
def show_player_move_popup():
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    messagebox.showinfo("Your Move", "It's your turn to move!")
    root.destroy()
    
def show_computer_move_popup():
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    messagebox.showinfo("Computer Move", "The computer is making its move.")
    root.destroy()

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
    detector = ElevatorPanelDetector(model_path, class_names, target_class)

    #run command if spacebar is pressed
    play_game(detector)
    
    # print("Press spacebar to run the detector")
    # def on_press(key):
    #     if key == keyboard.Key.space:
    #         a = detector.process_frames()
    #         for row in a:
    #             print(row)
    #         # for row in b:
    #         #     print(row)
    #         #return False  # Stop listener

    # while True:
    #     with keyboard.Listener(on_press=on_press) as listener:
    #         listener.join() 
    

if __name__ == "__main__":
    main()