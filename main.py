import cv2
import numpy as np
from PIL import Image
import os
import argparse

class FrameProcessor:
    def __init__(self):
        self.frame_image = None
        self.selected_points = []
        self.current_point = None
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for selecting corners of the frame area."""
        if event == cv2.EVENT_LBUTTONDOWN and len(self.selected_points) < 4:
            self.selected_points.append((x, y))
            # Draw point
            cv2.circle(self.frame_image, (x, y), 5, (0, 255, 0), -1)
            if len(self.selected_points) > 1:
                # Draw line between last two points
                cv2.line(self.frame_image,
                        self.selected_points[-2],
                        self.selected_points[-1],
                        (0, 255, 0), 2)
            cv2.imshow('Frame Selection', self.frame_image)

    def select_frame_area(self, frame_path):
        """Open the frame image and let user select the corners of the frame area."""
        if not os.path.exists(frame_path):
            raise FileNotFoundError(f"Frame image not found at: {frame_path}")
            
        self.frame_image = cv2.imread(frame_path)
        if self.frame_image is None:
            raise ValueError(f"Failed to load image at: {frame_path}")
            
        clone = self.frame_image.copy()
        cv2.namedWindow('Frame Selection')
        cv2.setMouseCallback('Frame Selection', self.mouse_callback)

        print("Click the four corners of the frame area (clockwise from top-left)")
        print("Press 'r' to reset selection")
        print("Press 'c' to continue when done")

        while True:
            cv2.imshow('Frame Selection', self.frame_image)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('r'):  # Reset
                self.frame_image = clone.copy()
                self.selected_points = []
                print("Selection reset. Start clicking corners again.")
            elif key == ord('c'):  # Continue when done
                if len(self.selected_points) == 4:
                    break
                else:
                    print(f"Please select all 4 corners. Currently selected: {len(self.selected_points)}")

        cv2.destroyAllWindows()
        return np.array(self.selected_points, dtype=np.float32)

    def embed_image(self, frame_path, image_to_embed_path, output_path):
        """Embed a new image into the marked frame area."""
        # Verify files exist
        if not os.path.exists(frame_path):
            raise FileNotFoundError(f"Frame image not found at: {frame_path}")
        if not os.path.exists(image_to_embed_path):
            raise FileNotFoundError(f"Image to embed not found at: {image_to_embed_path}")
            
        # Read the frame image
        frame = cv2.imread(frame_path)
        if frame is None:
            raise ValueError(f"Failed to load frame image at: {frame_path}")
            
        # Read and resize the image to embed
        img_to_embed = cv2.imread(image_to_embed_path)
        if img_to_embed is None:
            raise ValueError(f"Failed to load image to embed at: {image_to_embed_path}")
        
        # Get destination points (rectangle corners in clockwise order)
        dst_points = np.array([
            [0, 0],
            [img_to_embed.shape[1], 0],
            [img_to_embed.shape[1], img_to_embed.shape[0]],
            [0, img_to_embed.shape[0]]
        ], dtype=np.float32)
        
        # Calculate perspective transform
        matrix = cv2.getPerspectiveTransform(dst_points, self.selected_points)
        
        # Warp the image to embed
        warped = cv2.warpPerspective(
            img_to_embed,
            matrix,
            (frame.shape[1], frame.shape[0])
        )
        
        # Create a mask for the warped image
        mask = np.zeros(frame.shape, dtype=np.uint8)
        points = self.selected_points.astype(np.int32)
        cv2.fillPoly(mask, [points], (255, 255, 255))
        
        # Combine the images
        frame_bg = cv2.bitwise_and(frame, cv2.bitwise_not(mask))
        final = cv2.add(frame_bg, warped)
        
        # Save the result
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        cv2.imwrite(output_path, final)
        print(f"Saved result to: {output_path}")

def parse_arguments():
    parser = argparse.ArgumentParser(description='Embed an image into a frame photo.')
    parser.add_argument('-frame', required=True, help='Path to the frame photo')
    parser.add_argument('-embed', required=True, help='Path to the image to embed')
    parser.add_argument('-output', required=True, help='Path for the output image')
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Initialize the processor
    processor = FrameProcessor()
    
    # Process the frame and embed the image
    points = processor.select_frame_area(args.frame)
    processor.selected_points = points
    processor.embed_image(args.frame, args.embed, args.output)

if __name__ == "__main__":
    main()
