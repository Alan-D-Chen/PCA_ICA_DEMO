# PCA_ICA_DEMO
Demo for PCA(Principal Component Analysis) &amp; ICA(Independent Component Analysis) in data analysis in Python and image separation written in MATLAB 

>>This is a simple demo for PCA and ICA in Python 3.6.
The powerpoint we made just for lesson , not for business or money at all.

_*PCA:*_
I perfer to run it in Jupyter notebook.If you like, you can change the path for input dataset by rewrite 
'''
df=pd.read_excel('E:\pycharm-items-github\grade3.xls')
'''
in line 12.

_*ICA:*_
Step 1 show a PICs for the orignal lines, 4 lines for different data;
Srep 2 A PICs named mixed_lines for mixing 4 lines for different data;
Step 3 show the separated lines in a PIC called After_lines.jpy.



MATLAB:
Just run ICA.m to see the results, you can change the imread parameter to read different pic files and change mix matrix A to see the difference.
For an online video lesson, please refer: Youtube lesson.
For the original paper of ICA, please refer: paper.
If you use pic1.jpg and pic2.jpg and the mix matrix is [0.8 0.2; 0.2 0.8], then the result will be:

If you use pic3.jpg and pic4.jpg and the mix matrix is [0.8 0.2; 0.2 0.8], then the result will be:

If you use pic4.jpg and pic5.jpg and the mix matrix is [0.8 0.2; 0.2 0.8], then the result will be:

Good Luck.
2019, TJ.
