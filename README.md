# exercise01
debugging exercise1
This exercise focuses on identifying and fixing a logical bug in a Python function that maps a numeric ID to a fruit name. The debugging process required understanding Python data structures and correcting incorrect assumptions about element ordering.
problem:
The method id_to_fruit() trys to return a fruit from the set based on its index (fruit_id). However, the function is returning incorrect results for the given indices. This is because sets in Python are unordered collections, which means there’s no predictable order for the elements inside a set and data structure is wrong. In Python, sets are unordered:
the returned results do not matched outputs:

Index 1 → "orange" 

Index 3 → "kiwi" 

Index 4 → "strawberry" 

Solution:

To fix this issue, we need to sort the set into a list before indexing it. Sorting guarantees a deterministic order and allows us to safely access fruits by their index.
Find the debugging version as Exerciseone.py file.

EXERCISE2:

Problem:
The function is expected to return a new NumPy array where the x and y coordinates are flipped for every bounding box. However, the returned result is incorrect.
the original implementation contains:
coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3], = coords[:, 1], coords[:, 1], coords[:, 3], coords[:, 2]
this code has two problems:

coords[:, 1] is assigned twice, meaning column 0 and column 1 receive the same values.

The intended swap between (x1, y1) and (x2, y2) is not performed correctly.
find the debugging version as exercise2.py file

EXERCISE3

Problem:

The function reads precision-recall values from a CSV file and plots them. However, if we compare the plot with the values present in the CSV file, it is found that the points on the plot do not correspond to the actual coordinates. This is an indication of a logical error.
Issue 1: Precision and Recall Are Plotted in the Wrong Order
precision, recall
the plotting code uses:
plt.plot(results[:, 1], results[:, 0])

The function description

The axis labels

The expected behavior

Issue 2: Data Is Read as Strings Instead of Numbers:

The CSV reader returns each row as a list of strings. After the stacking operation, the NumPy array still contains string values.
This may lead to incorrect behavior of the plots or implicit type conversion.

Solution:
1.Explicitly converting CSV values to floats

2.Plotting precision on the x-axis and recall on the y-axis

3.Aligning axis labels with plotted data
find the debugging version as Exercise3(1).py file

Exercise4:

problems:
When the batch_size is changed from 32 to 64, a ValueError occurs:
ValueError: "Using a target size (torch.Size([128, 1])) that is different to the input size (torch.Size([96, 1])) is deprecated. Please ensure they have the same size."

This is due to the fact that the batch size influences the shape of the real and generated samples. If the batch size is not properly taken care of, as in the loss calculation steps, it will lead to a mismatch in the tensor sizes.
Solution:
we should ensure that either the real samples and generated samples are handled, Particularly, the loss calculations should properly handle the batch size, and the size of real_samples_labels and generated_samples_labels should match the batch size.

Cosmetic Bug:

problem:
The Discriminator and Generator loss are printed, but the displayed figure of the output (i.e., the generated images) is not clear or is displayed incorrectly.
solution: 
we should ensure that the losses are correctly displayed and formatted, and the output figure is updated correctly.

find the debugging version as exercise4.py file 
