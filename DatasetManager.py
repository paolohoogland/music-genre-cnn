from PIL import Image

class DatasetManager:
    def normalize_image(self, image_path, size=(256, 256)):
        img = Image.open(image_path)
        return img.resize(size)

    def save_image(self, image, output_path):
        image.save(output_path)
