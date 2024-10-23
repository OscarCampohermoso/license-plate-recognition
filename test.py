import cv2 as cv
import matplotlib.pyplot as plt
import string
import easyocr
from ultralytics import YOLO
# Mapping dictionaries for character conversion
# characters that can easily be confused can be
# verified by their location - an `O` in a place
# where a number is expected is probably a `0`
dict_char_to_int = {'O': '0',
                    'I': '1',
                    'J': '3',
                    'A': '4',
                    'G': '6',
                    'S': '5'}

dict_int_to_char = {'0': 'O',
                    '1': 'I',
                    '3': 'J',
                    '4': 'A',
                    '6': 'G',
                    '5': 'S'}

coco_model = YOLO('yolov8s.pt')  # Pre-trained YOLO model for car detection
np_model = YOLO('./runs/detect/train/weights/best.pt')

# Initialize OCR reader
reader = easyocr.Reader(['en'])  # Specify the language for OCR
def license_complies_format(text):
    print("Checking license plate format:")
    print(text)

    # Check for 5 or 7 characters (3 or 4 digits + 2 or 3 letters)
    if len(text) not in [5, 7]:
        print("License plate format is incorrect.")
        return False

    # Split the text into digits and letters based on its length
    if len(text) == 5:
        digits_part = text[:3]  # First 3 digits
        letters_part = text[3:]  # Last 2 letters
    else:  # len(text) == 7
        digits_part = text[:4]  # First 4 digits
        letters_part = text[4:]  # Last 3 letters

    # Check if the digits part is valid
    for char in digits_part:
        if char not in '0123456789' and char not in dict_char_to_int.keys():
            print("License plate format is incorrect.")
            return False

    # Check if the letters part is valid
    for char in letters_part:
        if char not in string.ascii_uppercase and char not in dict_int_to_char.keys():
            print("License plate format is incorrect.")
            return False

    print("License plate format is correct.")
    print("License plate: ", text)
    return True


def format_license(text):
    """
    Format the Bolivian license plate by converting confused characters.
    The first part contains 3 or 4 digits, and the last part contains exactly 3 letters.
    Returns the formatted plate.
    """
    text = text.strip().upper()
    formatted_text = []
    length = len(text)

    # Convert the last 3 characters to letters if they are numbers
    for i in range(length - 1, length - 4, -1):
        if i >= 0 and text[i] in dict_int_to_char:
            formatted_text.append(dict_int_to_char[text[i]])
        elif i >= 0:
            formatted_text.append(text[i])

    # Convert the remaining characters to digits if they are letters
    for i in range(length - 4, -1, -1):
        if text[i] in dict_char_to_int:
            formatted_text.append(dict_char_to_int[text[i]])
        else:
            formatted_text.append(text[i])

    # Reverse the list to get the correct order
    formatted_text.reverse()

    # Combine formatted characters
    formatted_plate = ''.join(formatted_text)

    return formatted_plate

def read_license_plate(license_plate_crop):
    # Perform OCR on the cropped license plate image
    detections = reader.readtext(license_plate_crop)
    print(detections)

    for detection in detections:
        bbox, text, score = detection

        # Process the text: convert to uppercase and remove spaces
        text = text.upper().replace(' ', '')
        print("Text: ", text, "Score: ", score)
        
        #Verify if the detected text matches a valid license plate format
        if license_complies_format(text):
            # If valid, return the formatted license plate and its OCR score
            print("Valid license plate detected: ", text)
            return format_license(text), score

    # If no valid license plate is found, return None
    return None, None

# Load the static image
image_path = './img/6009PLU.jpg'
image = cv.imread(image_path)

# Ensure tracking logic is bypassed
coco_model.callbacks = {}  # Remove any callbacks related to tracking

# Detect vehicles in the static image
vehicles = [2, 3, 5, 7]  # Define vehicle class IDs (e.g., cars, trucks)
results = {}

# Vehicle detector (using predict instead of track)
detections = coco_model.predict(image)[0]  # Make sure this is just detection, no tracking
vehicle_bounding_boxes = []

for detection in detections.boxes.data.tolist():
    x1, y1, x2, y2, score, class_id = detection
    print(score, class_id)
    if int(class_id) in vehicles and score > 0.5:
        vehicle_bounding_boxes.append([x1, y1, x2, y2, None, score])  # track_id is None
        # Crop the vehicle's region of interest (ROI)
        roi = image[int(y1):int(y2), int(x1):int(x2)]

        # License plate detector for the region of interest
        license_plates = np_model.predict(roi)[0]

        # Process license plate
        for license_plate in license_plates.boxes.data.tolist():
            plate_x1, plate_y1, plate_x2, plate_y2, plate_score, _ = license_plate

            # Crop the plate from the region of interest
            plate = roi[int(plate_y1):int(plate_y2), int(plate_x1):int(plate_x2)]

            # Convert the plate to grayscale
            plate_gray = cv.cvtColor(plate, cv.COLOR_BGR2GRAY)

            # OCR to read the license plate text
            np_text, np_score = read_license_plate(plate_gray)
            print(np_text, np_score)
            # If plate is readable, store results
            if np_text is not None:
                results[None] = {  # track_id is None
                    'car': {
                        'bbox': [x1, y1, x2, y2],
                        'bbox_score': score
                    },
                    'license_plate': {
                        'bbox': [plate_x1, plate_y1, plate_x2, plate_y2],
                        'bbox_score': plate_score,
                        'number': np_text,
                        'text_score': np_score
                    }
                }

                # Draw vehicle and license plate bounding boxes on the image
                cv.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                cv.rectangle(roi, (int(plate_x1), int(plate_y1)), (int(plate_x2), int(plate_y2)), (0, 255, 0), 2)
                cv.putText(roi, np_text, (int(plate_x1), int(plate_y1 - 10)), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

# Show the results in a window
cv.imshow('License Plate Detection', image)
cv.waitKey(0)
cv.destroyAllWindows()
