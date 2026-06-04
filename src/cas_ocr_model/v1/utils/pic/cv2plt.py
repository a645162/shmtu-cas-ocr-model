import cv2
import matplotlib.pyplot as plt


def show_opencv_img_by_plt(image):
    # 将BGR格式转换为RGB格式
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 使用Matplotlib显示图像
    plt.imshow(image_rgb)
    plt.axis('off')  # 关闭坐标轴
    plt.show()


if __name__ == '__main__':
    # 示例用法
    # 读取图像
    image_path = 'example.jpg'
    image = cv2.imread(image_path)

    # 显示图像
    show_opencv_img_by_plt(image)
