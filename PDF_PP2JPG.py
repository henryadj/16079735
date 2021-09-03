import glob
import os
import win32com.client
from pdf2image import convert_from_path

class converter:
    
    def __init__(self,file_directory,image_directory):
        """""""""
        Converts Powerpoint slides to PDF, and PDF documents into JPG images

        file_directory : where the PDF and PowerPoint Slides are located

        image_directory : The output directory for the JPG images

        """""""""
        self.dir = file_directory

        # Creates the image directory if it doesn't exist
        if not os.path.exists(image_directory):
            os.mkdir(image_directory)

        self.img_dir = image_directory
    
    
    def pdf2jpg(self,dpi = 300):
        """""""""
        dpi = The quality of the exported image, higher is better but takes longer to convert
        
        Output : Saves each individual image in the image directory as (xx_0.jpg, xx_1.jpg, ...) 
        **NOTE : Please do change the PATH for poppler in the convert_from_path argument**
        """""""""
        for pdf in glob.glob(os.path.join(self.dir,"*.pdf")):
            #Converts each individual page in a pdf to images
            pages = convert_from_path(pdf ,dpi , poppler_path = r"C:\Users\hang026\poppler-21.03.0\Library\bin") # Ensure poppler path is correct
            
            # Getting the document name 
            document_name = pdf.split('\\')[-1][:-4]

            for page in range(len(pages)):
                
                # Save to image directory with the same name as the document but with .jpg at the end instead of .pdf
                pages[page].save(self.img_dir +'\\' +  document_name + "_{}".format(page) + ".jpg",'JPEG') # save to directory
                
    def PPTX2PDF(self, filename,formatType = 32):
        powerpoint = win32com.client.Dispatch("Powerpoint.Application")
        powerpoint.Visible = 1

        newname = os.path.splitext(filename)[0] +".pdf"
        deck = powerpoint.Presentations.Open(filename)
        deck.SaveAs(newname, formatType)
        deck.Close()
        powerpoint.Quit()
        
    def Convert_All(self, dpi =300):
        
        # Check if PPTX already converted
        PPTX_file = glob.glob(os.path.join(self.dir,"*.pptx"))
        
        for pp in PPTX_file :
            
            
            document_name = pp.split('\\')[-1][:-5]
            
            # Check if PDF version exist
            if not os.path.exists(self.dir +'\\' + document_name+'.pdf'):
                # Convert PPTX to PDF
                self.PPTX2PDF(pp)
                
        self.pdf2jpg(dpi)
                
            
        
        
        
        
