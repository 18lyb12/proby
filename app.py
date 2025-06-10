import os
import threading
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify
from rdkit import Chem
from rdkit.Chem import Draw

from proby.app.pipeline import process_files
from proby.app.util import delete_files_in_folder, shared_logger

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'input'
app.config['OUTPUT_FOLDER'] = 'output'
log_messages = []


# Function to log messages
def log_message(message):
    log_messages.append(message)
    if len(log_messages) > 100:
        log_messages.pop(0)


# Route to handle homepage
@app.route('/')
def home():
    return render_template('index.html')


# Route to handle Page 1
@app.route('/page1', methods=['GET', 'POST'])
def page1():
    if request.method == 'POST':
        if 'files[]' in request.files:
            # delete previously uploaded files in input folder
            delete_files_in_folder(app.config['UPLOAD_FOLDER'])

            files = request.files.getlist('files[]')
            for file in files:
                filename = file.filename
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                shared_logger.log(f'Uploaded file: {filename}')

            # Start processing in a separate thread
            metadata = {
                "input_data_folder": app.config['UPLOAD_FOLDER'],
                "app_output_data_folder": app.config['OUTPUT_FOLDER']
            }
            threading.Thread(target=process_files, args=(metadata,)).start()
            return redirect(url_for('page1'))

    output_files = os.listdir(app.config['OUTPUT_FOLDER'])
    return render_template('page1.html', output_files=output_files)


# Route to fetch the real-time log
@app.route('/get_log', methods=['GET'])
def get_log():
    return jsonify(shared_logger.get_logs())


# Route to download output files
@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)


# Route to handle Page 2
@app.route('/page2')
def page2():
    return render_template('page2.html')


# Route to handle Page 2
@app.route('/page3', methods=['GET', 'POST'])
def page3():
    if request.method == 'POST':
        text_input = request.form['text_input']
        try:

            smiles_list = text_input.split(",")

            failed_smiles = [smiles for smiles in smiles_list if not Chem.MolFromSmiles(smiles)]
            if failed_smiles:
                message = f"{', '.join(failed_smiles)} {'is' if len(failed_smiles) == 1 else 'are'} invalid smiles. Please enter valid smiles separated by a comma."
                raise ValueError(message)

            img = Draw.MolsToGridImage([Chem.MolFromSmiles(smiles) for smiles in smiles_list],
                                       molsPerRow=2, subImgSize=(500, 500), legends=smiles_list, returnPNG=False)

            # Save image to static folder
            image_path = os.path.join('static', 'images', 'smiles.png')
            img.save(image_path)

            message = f"Smiles Plot"
            return render_template('page3.html', message=message,
                                   image_url=url_for('static', filename='images/smiles.png'))
        except ValueError:
            return render_template('page3.html',
                                   message=message)

    return render_template('page3.html')


# Route to go back to the homepage
@app.route('/back_to_home')
def back_to_home():
    return redirect(url_for('home'))


if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
    os.makedirs('static/images', exist_ok=True)
    delete_files_in_folder(app.config['UPLOAD_FOLDER'])
    delete_files_in_folder(app.config['OUTPUT_FOLDER'])
    app.run(debug=True)
