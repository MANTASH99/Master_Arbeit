import matplotlib.pyplot as plt

# Sample data
frequencies1 = ['250', '315', '400', '500', '630', '800', '1000', '1250', '1600',
               '2000', '2500', '3150', '4000', '5000', '6300', '8000' ,'10000']
frequencies2 = ['125','160' , '200', '250', '315', '400', '500', '630', '800', '1000', '1250', '1600',
               '2000', '2500', '3150', '4000', '5000', '6300', '8000' , '10000']

frequencies =['250',	'500'	,'750',	'1000',	'1500',	'2000',	'3000',	'4000',	'6000',	'8000']
len_fre = len(frequencies1)
# Define specific important frequencies for each group

# Convert frequencies to their corresponding indices in the original list
group1_indices = [3,5]
group2_indices = [3,7]
group3_indices = [5,7]
#group1_indices_extra = [4,5]


group1_indices_bar = [3,4,5,7,9]
group2_indices_bar= [3,4,5,6,7]
group3_indices_bar = [1,3,5,6,7]

# Plotting
plt.figure(figsize=(12, 6))

# Plot colored areas for important frequencies for each group
plt.fill_between(group1_indices, 0, 1, color='red', alpha=0.3)
#plt.fill_between(group1_indices_extra, 0, 1, color='red', alpha=0.3)
plt.fill_between(group2_indices, 1.5, 2.5, color='blue', alpha=0.3)
plt.fill_between(group3_indices, 3, 4, color='green', alpha=0.3)
#plt.fill_between(group1_indices_extra, 0, 1, color='red', alpha=0.3)



# Plot all frequencies
plt.plot(range(1, len_fre + 1), [1]*len_fre, 'r--', alpha=0.3)
plt.plot(range(1, len_fre + 1), [2]*len_fre, 'b.', alpha=0.3)
plt.plot(range(1, len_fre + 1), [3]*len_fre, 'g--', alpha=0.3)
plt.bar(group3_indices_bar, [1], color='green', alpha=0.8,width=0.3 ,label='NAL 80db',bottom=3 )

plt.bar(group2_indices_bar, [1], color='blue', alpha=0.8,width=0.3 ,label='NAL 65db',bottom=1.5 )

plt.bar(group1_indices_bar, [1], color='red', alpha=0.8,width=0.3 ,label='NAL 50db',bottom=0 )
#
#für NAl 50 ylimt also 1 and otttum 0
#für Nal 80 db limt y 1 , butom 3
# Customize the plot
plt.xlabel('Frequencies (Hz)')
plt.ylabel('NAL')
plt.title('important frequencies NAL')
plt.xticks(range(1, len_fre + 1), frequencies1, rotation=45)
plt.yticks([0.5, 2 ,3.5], ['NAL 50db', 'NAL 65db' ,'NAL 80db' ])
plt.ylim(0, 4)
plt.legend(handlelength=0.7)
plt.grid(True)

# Show plot
plt.tight_layout()
plt.savefig('NAL111.png')
plt.show()