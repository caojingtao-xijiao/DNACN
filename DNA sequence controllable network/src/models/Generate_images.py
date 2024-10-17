def generate_images(test_data):

    image_tensor = self.data_set.data[top]
    image = Image.fromarray(image_tensor.numpy())
    image.save(os.path.join(save_dir, f'{top}.jpg'))