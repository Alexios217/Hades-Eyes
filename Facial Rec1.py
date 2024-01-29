import argparse
import pickle
from collections import Counter
from pathlib import Path

import face_recognition
from PIL import Image, ImageDraw

DEFAULT_ENCODINGS_PATH = Path("output/encodings.pkl")
BOUNDING_BOX_COLOR = "blue"
TEXT_COLOR = "white"

parser = argparse.ArgumentParser(description="Recognize faces in an image")
parser.add_argument("--train", action="store_true", help="Train on input data")
parser.add_argument(
    "--validate", action="store_true", help="Validate trained model"
)
parser.add_argument(
    "--test",
    type=str,
    help="Test the trained model on a specific image"
)
args = parser.parse_args()

def encode_known_faces(model="hog"):
    # Load images from the training directory, detect faces, and encode them
    # Save the encoded faces and their corresponding labels to a pickle file
    pass

def recognize_faces(image_location, model="hog"):
    with open(DEFAULT_ENCODINGS_PATH, "rb") as f:
        loaded_encodings = pickle.load(f)

    input_image = face_recognition.load_image_file(image_location)

    input_face_locations = face_recognition.face_locations(
        input_image, model=model
    )
    input_face_encodings = face_recognition.face_encodings(
        input_image, input_face_locations
    )

    for bounding_box, unknown_encoding in zip(
        input_face_locations, input_face_encodings
    ):
        name = _recognize_face(unknown_encoding, loaded_encodings)
        if not name:
            name = "Unknown"
        _display_face(bounding_box, name)

def _recognize_face(unknown_encoding, loaded_encodings):
    boolean_matches = face_recognition.compare_faces(
        loaded_encodings["encodings"], unknown_encoding
    )
    votes = Counter(
        name
        for match, name in zip(boolean_matches, loaded_encodings["names"])
        if match
    )
    if votes:
        return votes.most_common(1)[0][0]

def _display_face(bounding_box, name):
    top, right, bottom, left = bounding_box
    pillow_image = Image.fromarray(input_image)
    draw = ImageDraw.Draw(pillow_image)
    text_left, text_top, text_right, text_bottom = draw.textbbox(
        (left, bottom), name
    )
    draw.rectangle(((left, top), (right, bottom)), outline=BOUNDING_BOX_COLOR)
    draw.rectangle(
        ((text_left, text_top), (text_right, text_bottom)),
        fill="blue",
        outline="blue",
    )
    draw.text(
        (text_left, text_top),
        name,
        fill="white",
    )
    pillow_image.show()

def validate(model="hog"):
    for filepath in Path("validation").rglob("*"):
        if filepath.is_file():
            recognize_faces(
                image_location=str(filepath.absolute()), model=model
            )

if __name__ == "__main__":
    if args.train:
        encode_known_faces()
    elif args.validate:
        validate()
    elif args.test:
        recognize_faces(args.test)
    else:
        print("No arguments provided. Use --help for more information.")