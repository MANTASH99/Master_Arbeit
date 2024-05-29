from PIL import Image

# Open the individual image files
image1 = Image.open("loss.png")
image2 = Image.open("loss2.png")
image3 = Image.open("loss3.png")
image4 = Image.open("loss4.png")
image5 = Image.open("loss5.png")
image6 = Image.open("loss6.png")

# Get the dimensions of the images
width1, height1 = image1.size
width2, height2 = image2.size
width3, height3 = image3.size
width4, height4 = image4.size
width5, height5 = image5.size
width6, height6 = image6.size

# Calculate the total width and height of the merged image
total_width = max(width1 + width4, width2 + width5, width3 + width6)
total_height = max(height1, height2, height3) * 3

# Create a new blank image with the combined dimensions
merged_image = Image.new("RGB", (total_width, total_height), (255, 255, 255))  # White background

# Paste the individual images onto the new blank image
merged_image.paste(image1, (0, 0))
merged_image.paste(image2, (0, height1))
merged_image.paste(image3, (0, height1 * 2))

merged_image.paste(image4, (width1, 0))
merged_image.paste(image5, (width1, height1))
merged_image.paste(image6, (width1, height1 * 2))

# Save the merged image
merged_image.save("merged_loss_figures.png")

# Display the merged image
merged_image.save('losses.png')
merged_image.show()
