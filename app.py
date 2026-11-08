from flask import Flask ,render_template, send_from_directory
from flask_wtf import FlaskForm 
import os


from utils.model import VGGEncoder,Decoder,adaptive_IN
from werkzeug.utils import secure_filename
from flask_wtf.file import FileAllowed,FileRequired,FileField
from wtforms import HiddenField, SubmitField,FloatField
from flask_bootstrap import Bootstrap
from PIL import Image

from torchvision.utils import save_image
import torch
from torchvision import transforms

app=Flask(__name__)
app.config['SECRET_KEY']=os.environ.get('SECRET_KEY','adain-nst-key')
app.config['UPLOAD_FOLDER']='static/uploads'
Bootstrap(app)

os.makedirs(app.config['UPLOAD_FOLDER'],exist_ok=True)

class Form(FlaskForm):
    content=FileField("Content Image",validators=[FileRequired(message='Please select the content image'),FileAllowed(['jpg','png','jpeg'],message='Selecte image must be in jpg,png,jpeg format')])
    style=FileField("Style Image",validators=[FileRequired(message='Please select the style image'),FileAllowed(['jpg','png','jpeg'],message='Selecte image must be in jpg,png,jpeg format')])
    alpha=FloatField("Alpha",default=1.0)
    content_path=HiddenField()
    style_path=HiddenField()
    submit=SubmitField('Transfer Style')

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoder=VGGEncoder('vgg_normalised.pth')
decoder=Decoder()
encoder.to(device)
decoder.to(device)
decoder.load_state_dict(torch.load('C:/Users/91917/OneDrive/Desktop/AI_ML lec content/Major Projects/NST/Adain-Style-Transfer/experiment/training/decoder 4 (1).pth',map_location=device))

encoder.eval()
decoder.eval()

def style_transfer(content_img,style_img,encoder,decoder,device,alpha=1.0):
    transformation=transforms.Compose([transforms.Resize(512),
                        transforms.ToTensor()
                        ])

    content_tensor=transformation(content_img).unsqueeze(0).to(device)
    style_tensor=transformation(style_img).unsqueeze(0).to(device)

    with torch.no_grad():
        content_feats=encoder(content_tensor,is_test=True)
        style_feats=encoder(style_tensor,is_test=True)

        fg=adaptive_IN(content_feats,style_feats)

        # Adding style streangth to generated image 
        final_img_feats=alpha*fg+(1-alpha)*content_feats

        generated_image=decoder(final_img_feats)
        

        return generated_image

@app.route('/', methods=['GET', 'POST'])
def home_page():
    form=Form()
    result_filename=None
    error=None
    style_filename=None
    content_filename=None
    

    if form.validate_on_submit():
        
        if form.content.data and form.content.data.filename:
           content_filename=secure_filename( form.content.data.filename)
           form.content.data.save(os.path.join(app.config['UPLOAD_FOLDER'],content_filename))
           form.content_path.data=content_filename
        else:
           content_filename=form.content_path.data


        if form.style.data and form.style.data.filename:     
           style_filename=secure_filename(form.style.data.filename)
           form.style.data.save(os.path.join(app.config['UPLOAD_FOLDER'],style_filename))
           form.style_path.data=style_filename
        else:
           style_filename=form.style_path.data

        

        if style_filename and content_filename:
            image_path=app.config['UPLOAD_FOLDER']
            content_path=os.path.join(image_path,content_filename)
            style_path=os.path.join(image_path,style_filename)
            
            try:
                content_img=Image.open(content_path).convert('RGB')
                style_img=Image.open(style_path).convert('RGB')
                alpha = form.alpha.data
                if alpha is None:
                    alpha = 1.0
                alpha=float(alpha)
                result_image=style_transfer(content_img,style_img,encoder,decoder,device,alpha)
                content_name=os.path.splitext(content_filename)[0]
                style_name=os.path.splitext(style_filename)[0]
                result_filename =f"generated_{content_name}_{style_name}.png"
                result_path=os.path.join(image_path,result_filename)
                save_image(result_image.to(device).clamp(0,1),result_path)
                
            except Exception as e:
                error=str(e)

        else:
            if not style_filename :
                error="Please upload Style Image"
            if not content_filename :
                        error="Please upload Content Image"
    return render_template('index.html',form=form,content_image=content_filename,style_image=style_filename,result_image=result_filename,error=error)


@app.route('/examples/<path:filename>')
def send_example(filename):
    return send_from_directory("examples",filename)

@app.route('/static/uploads/<path:filename>')
def send_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],filename)
    


if __name__=="__main__":
    app.run(host="localhost",debug=True,port=5000)