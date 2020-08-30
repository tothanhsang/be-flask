import base64

# base64_img = ""

# base64_img_bytes = base64_img.encode('utf-8')
# with open('image.png', 'wb') as file_to_save:
#     decoded_image_data = base64.decodebytes(base64_img_bytes)
#     file_to_save.write(decoded_image_data)

# import base64
# with open("biden.jpg", "rb") as img_file:
#     my_string = base64.b64encode(img_file.read()).decode('utf-8')
# print(my_string)

# # import base64
 
# # def get_base64_encoded_image(image_path):
# #     with open(image_path, "rb") as img_file:
# #         return base64.b64encode(img_file.read()).decode('utf-8')

import base64
# image = open('thuan.jpg', 'rb')
# image_read = image.read()
# image_64_encode = base64.encodebytes(image_read)
input_file = open('input.txt', 'r')
coded_string = input_file.read()
image_64_decode = base64.b64decode(coded_string)
image_result = open('decode.jpg', 'wb') 
image_result.write(image_64_decode)