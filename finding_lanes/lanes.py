# import cv2 as cv2# will import opencv
# import numpy as np
# import matplotlib.pyplot as plt


# def make_points(image, line):
#     """return the points of the straight line in an array"""
    
#     slope, intercept = line
#     y1 = int(image.shape[0])# bottom of the image
#     y2 = int(y1*3/5)         # slightly lower than the middle
#     x1 = int((y1 - intercept)/slope)
#     x2 = int((y2 - intercept)/slope)
    
#     return np.array([x1, y1, x2, y2])


# def average_slope_intercept(image, lines):
#     """averages the line so instead of multiple straight lines we get a single line on both sides"""
    
#     left_fit    = []
#     right_fit   = []
    
#     if lines is None:
#         return None
    
#     for line in lines:
#         x1, y1, x2, y2 = line.reshape(4)
#         fit = np.polyfit((x1,x2), (y1,y2), 1)
#         slope = fit[0]
#         intercept = fit[1]
#         if slope < 0: # y is reversed in image
#             left_fit.append((slope, intercept))
#         else:
#             right_fit.append((slope, intercept))
#     # add more weight to longer lines
    
#     left_fit_average  = np.average(left_fit, axis=0)
#     right_fit_average = np.average(right_fit, axis=0)
#     left_line  = make_points(image, left_fit_average)
#     right_line = make_points(image, right_fit_average)
    
#     return np.array([left_line, right_line])


# def canny(image):
#     """Takes an image as input converts it into grayscale calculates its Gaussian blur and 
#     Uses that to show its edges using canny edge detection."""
    
#     gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
#     blur = cv2.GaussianBlur(gray, (5, 5), 0)
#     canny = cv2.Canny(blur, 50, 100)

#     return canny


# def display_lines(image, lines):
#     """It displays the line generated using the hough transform"""
    
#     line_image = np.zeros_like(image)
#     if lines is not None:
#         for line in lines:
#             x1, y1, x2, y2 = line.reshape(4)
#             cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    
#     return line_image


# def region_of_interest(image):
#     """Takes an image as input returns the area of the image that is required by masking the image
#     and using bitwise operation and getting the required area in the image."""
    
#     height = image.shape[0]
#     polygons = np.array([
#         [(200, height ), (1100, height), (550, 250)]
#         ])
#     mask = np.zeros_like(image)
#     cv2.fillPoly(mask, polygons, 255)
#     masked_image = cv2.bitwise_and(image, mask)
    
#     return masked_image


# # image = cv2.imread("D:\\projects that i am doing\\self_driving_car\\finding_lanes\\test_image.jpg")
# # lane_image = np.copy(image)
# # canny_image = canny(lane_image)
# # cropped_image = region_of_interest(canny_image)
# # lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=4, maxLineGap=5 ) # Using hough lines to find the point where the line intersects the most and finding straight lines
# # averaged_lines = average_slope_intercept(image, lines)
# # line_image = display_lines(lane_image, averaged_lines)
# # combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)# combines the lines and the image with the weight of the image as 0.8 to darken the pixels and add the lines and images to get the combo image.
# # cv2.imshow("result", combo_image)
# # cv2.waitKey(0)


# cap = cv2.VideoCapture("D:\\projects that i am doing\\self_driving_car\\finding_lanes\\test2.mp4")
# while(cap.isOpened()):
#     _, frame = cap.read()
#     canny_image = canny(frame)
#     cropped_canny = region_of_interest(canny_image)
#     lines = cv2.HoughLinesP(cropped_canny, 2, np.pi/180, 100, np.array([]), minLineLength=40,maxLineGap=5)
#     averaged_lines = average_slope_intercept(frame, lines)
#     line_image = display_lines(frame, averaged_lines)
#     combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
#     cv2.imshow("result", combo_image)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# cap.release()
# # cv2.destroyAllWindows()







import cv2
import numpy as np
import matplotlib.pyplot as plt


def make_coord(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1*(2/3))
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([x1, y1, x2, y2])

def average_lines(image, lines):
    left = []
    right = []
    for line in lines:
        x1,y1,x2,y2 = line.reshape(4)
        parameters = np.polyfit((x1,x2),(y1,y2),1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left.append((slope, intercept))
        else:
            right.append((slope, intercept))
    left_avg = np.average(left, axis=0)
    right_avg = np.average(right, axis=0)
    left_line = make_coord(image, left_avg)
    right_line = make_coord(image, right_avg)
    return np.array([left_line, right_line])


def Canny(im):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(gray, 100, 200)
    return canny


def region_of_interest(im):
    height = im.shape[0]
    polygons = np.array([
    [(200,height),(1100, height),(550,250)]
    ])
    mask = np.zeros_like(im)
    cv2.fillPoly(mask, polygons, (255,255,255))
    masked_lane = np.bitwise_and(im, mask)
    return masked_lane

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line.reshape(4)
            cv2.line(line_image, (x1,y1), (x2,y2), (0,255,0), 5)
    return line_image

# Main program
# Capture the video
def canny():
    cap = cv2.VideoCapture("test2.mp4")
    while(cap.isOpened()):
        # Read the frame
        _, frame = cap.read()

        # Detect edges
        canny_image = Canny(frame)
        cv2.imshow("result", canny_image)
        # Identify the region of interest
        cropped_image = region_of_interest(canny_image)
        # Apply hough transform to identify lines
        lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength = 40, maxLineGap = 5)
        # Get a single left lane marker and right lane marker
        averaged_lines = average_lines(frame, lines)
        # Display the detected line
        line_image = display_lines(frame, averaged_lines)
        # Create a combination of original image and the detected lanes
        combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)

        # Display the detected lane for current frame
        
        # Wait 1 ms or till q is pressed on keyboard
        if cv2.waitKey(1) == ord('q'):
            break
    # Release the video        
    cap.release()
    cv2.destroyAllWindows()

def final():
    cap = cv2.VideoCapture("test2.mp4")
    while(cap.isOpened()):
        # Read the frame
        _, frame = cap.read()

        # Detect edges
        canny_image = Canny(frame)
        # Identify the region of interest
        cropped_image = region_of_interest(canny_image)
        # Apply hough transform to identify lines
        lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength = 40, maxLineGap = 5)
        # Get a single left lane marker and right lane marker
        averaged_lines = average_lines(frame, lines)
        # Display the detected line
        line_image = display_lines(frame, averaged_lines)
        # Create a combination of original image and the detected lanes
        combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)

        # Display the detected lane for current frame
        cv2.imshow("Detected_Lanes", combo_image)
        # Wait 1 ms or till q is pressed on keyboard
        if cv2.waitKey(1) == ord('q'):
            break
    # Release the video        
    cap.release()
    cv2.destroyAllWindows()



canny()
final()