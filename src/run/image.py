
import cv2
from scanner.tools import scanner
from scanner.tools import detector as detect

model = '../rtm_det_card_trainer.py'
weights = "../work_dirs/rtm_det_card_trainer/epoch_9.pth"
imagePath = '../media/cards3.jpg'
size = 1080
scoreThreshold = .5
save_image = False
mirror = False
include_flipped = True

def main():
    print("Starting process - inference with image")

    image_original = scanner.read_image(imagePath, size)
    image_copy = image_original.copy()

    # Initialize the DetInferencer
    detector = detect.Detector(model, weights)

    detections = detector.detect_objects(image_original, scoreThreshold)

    scanner.process_masks_to_cards(image_original, detections, mirror)
    scanner.hash_cards(detections, include_flipped)
    scanner.match_hashes(detections, include_flipped)

    scanner.draw_boxes(image_copy, detections)
    scanner.draw_masks(image_copy, detections)
    scanner.write_card_labels(image_copy, detections)

    # for detection in detections:
    #     if 'card_image' in detection:
    #         scanner.showImageWait(detection['card_image'])

    if save_image is True:
        # Save the image
        output_path = 'output_image.jpg'
        cv2.imwrite(f'output/{output_path}', image_copy)
        print('Image saved successfully.')

    scanner.show_image_wait(image_copy)


if __name__ == '__main__':
    main()
