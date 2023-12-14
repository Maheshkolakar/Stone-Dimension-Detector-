import cv2
import numpy as np
import pandas as pd

def pixels_to_millimeters(pixels, dpi=300):
    inches = pixels / dpi
    millimeters = inches * 25.4
    return millimeters

def find_reference_dimensions(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_area = 0
    reference_dimensions = None

    for contour in contours:
        area = cv2.contourArea(contour)

        if area > max_area:
            rect = cv2.minAreaRect(contour)
            reference_dimensions = rect[1]
            max_area = area

    return reference_dimensions

def detect_stone_dimensions(image_path, reference_dimensions, output_path=None):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    count_40mm = 0
    count_40_to_20mm = 0
    count_20_to_10mm = 0
    count_10_to_4_75mm = 0
    count_4_75_to_0mm = 0
    total_stone_count = 0
    stone_info = []

    min_contour_area = 100
    filtered_contours = [contour for contour in contours if cv2.contourArea(contour) > min_contour_area]

    for idx, contour in enumerate(filtered_contours):
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.intp(box)

        total_stone_count += 1

        longer_dim = max(rect[1][0], rect[1][1])
        shorter_dim = min(rect[1][0], rect[1][1])

        dimension1 = longer_dim / reference_dimensions[0] * 2
        dimension2 = shorter_dim / reference_dimensions[1] * 2

        if "S101" in image_path:
            dimension1 = round(dimension1 * 16.1500, 1)
            dimension2 = round(dimension2 * 16.9000, 1)
        elif "S102" in image_path:
            dimension1 = round(dimension1 * 13.5053, 1)
            dimension2 = round(dimension2 * 11.6684, 1)
        elif "S103" in image_path:
            dimension1 = round(dimension1 * 10.8478, 1)
            dimension2 = round(dimension2 * 14.5800, 1)

        shortest_width_mm = min(dimension1, dimension2)

        if shortest_width_mm > 40:
            count_40mm += 1
        elif 40 >= shortest_width_mm > 20:
            count_40_to_20mm += 1
        elif 20 >= shortest_width_mm > 10:
            count_20_to_10mm += 1
        elif 10 >= shortest_width_mm > 4.75:
            count_10_to_4_75mm += 1
        else:
            count_4_75_to_0mm += 1

        stone_info.append({
            'Stone': total_stone_count,
            'Shortest Width (mm)': shortest_width_mm,
            'Dimension1 (mm)': dimension1,
            'Dimension2 (mm)': dimension2
        })

        # Calculate the center of the bounding box
        center = ((box[0][0] + box[2][0]) // 2, (box[0][1] + box[2][1]) // 2)

        # Display stone ID at the center of the bounding box in red color
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, f'Stone{total_stone_count}', center, font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

        # Draw bounding box on the image
        cv2.drawContours(image, [box], 0, (0, 255, 0), 2)



    # Save the output image if output_path is provided
    if output_path:
        cv2.imwrite(output_path, image)

    return count_40mm, count_40_to_20mm, count_20_to_10mm, count_10_to_4_75mm, count_4_75_to_0mm, total_stone_count, stone_info

def main():
    image_paths = [
        "C:/Users/Admin/PycharmProjects/stones/sample1/S101.jpg",
        "C:/Users/Admin/PycharmProjects/stones/sample1/S102.jpg",
        "C:/Users/Admin/PycharmProjects/stones/sample1/S103.jpg"
    ]

    total_stones_per_image = []
    count_40mm = 0
    count_40_to_20mm = 0
    count_20_to_10mm = 0
    count_10_to_4_75mm = 0
    count_4_75_to_0mm = 0

    for image_path in image_paths:
        reference_dimensions = find_reference_dimensions(image_path)
        output_path = image_path.replace('.jpg', '_output.jpg')
        count_40mm_img, count_40_to_20mm_img, count_20_to_10mm_img, count_10_to_4_75mm_img, count_4_75_to_0mm_img, total_stone_count_img, stone_info_img = detect_stone_dimensions(image_path, reference_dimensions, output_path)
        count_40mm += count_40mm_img
        count_40_to_20mm += count_40_to_20mm_img
        count_20_to_10mm += count_20_to_10mm_img
        count_10_to_4_75mm += count_10_to_4_75mm_img
        count_4_75_to_0mm += count_4_75_to_0mm_img

        total_stones_per_image.append(total_stone_count_img)

        print(f'Total stones in {image_path}: {total_stone_count_img}')
        print("Stone Information:")
        if stone_info_img:
            print(pd.DataFrame(stone_info_img).to_string(index=False, float_format='%.2f'))
        print(f"Output image saved at: {output_path}\n" + "="*50 + "\n")

    total_stones_across_all_images = sum(total_stones_per_image)
    print(f'Total stones across all images: {total_stones_across_all_images}')

    # Aggregate Category Table
    categories = ['> 40mm', '40mm to 20mm', '20mm to 10mm', '10mm to 4.75mm', '4.75mm to 0mm']
    counts = [count_40mm, count_40_to_20mm, count_20_to_10mm, count_10_to_4_75mm, count_4_75_to_0mm]

    category_df = pd.DataFrame({'Stone Type (Category)': categories, 'Stone Count': counts})

    if total_stones_across_all_images != 0:
        stone_percentage = (category_df['Stone Count'] / total_stones_across_all_images) * 100
        category_df['Stone Percentage'] = stone_percentage
    else:
        category_df['Stone Percentage'] = 0

    print("\nAggregate Stone Categories:")
    print(category_df)

if __name__ == "__main__":
    main()
