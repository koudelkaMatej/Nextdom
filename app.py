from operator import index
import os
from tkinter.tix import FileSelectBox

from flask import Flask, request, redirect, session, url_for, abort,  make_response, render_template, send_file  #knihovna REST API
from flask.json import jsonify
from flask_sslify import SSLify
import pandas as pd
import openpyxl
from openpyxl import Workbook
from werkzeug.utils import secure_filename
from datetime import datetime

#graphs

import zipfile
import numpy as np
import json
import plotly
import plotly.express as px

from analyzeFromMatlab import *
from setOfThresholdsCreator import *

from apscheduler.schedulers.background import BackgroundScheduler
import logging
import shutil
import threading
import psutil

nextDOM = Flask(__name__)
sslify = SSLify(nextDOM)

nextDOM.config["UPLOAD_FOLDER"] = "static/uploadedFiles/"



@nextDOM.route('/', methods =["GET", "POST"])
def home():
    return render_template("/v2/index.html")

@nextDOM.route('/about', methods =["GET", "POST"])
def about():
    return render_template("/v2/about.html")


@nextDOM.route('/thresholdsnew', methods =["GET", "POST"])
def threshholdsnew():
    if request.method == "POST":
        # getting input with name = fname in HTML form
        loading = request.form.get("Healthy_Donors_From")
        NumberOfNormals = request.form.get("NumberOfNormals")
        saving = request.form.get("Save_as")
        dateOfStart = str(datetime.datetime.today().strftime('%d_%m_%Y_%H_%M'))
        if loading.__contains__("xlsx"):
            # getting input with name = lname in HTML form
            file = request.files['FileOfHealthyDonors']
            file.save(os.path.join(nextDOM.config['UPLOAD_FOLDER'], file.filename))
            filename = secure_filename(file.filename)
            #myfile = request.form.get("FileOfHealthyDonors")
            path = os.path.join(nextDOM.config['UPLOAD_FOLDER'], filename)

# doplnit co delat s XLSX
            return redirect(url_for('graphs', saving=saving, loading=loading, NumberOfNormals=NumberOfNormals, path=path))
        elif loading.__contains__("default"):
            return redirect(url_for('graphs', saving=saving, loading=loading, NumberOfNormals=NumberOfNormals))
        else:
            files = request.files.getlist('FileOfHealthyDonors')
            for file in files:
                filename = secure_filename(file.filename)
                if filename != '':
                    if not os.path.exists(nextDOM.config['UPLOAD_FOLDER'] + "/Threshold/" + dateOfStart + "/"):
                        os.makedirs(nextDOM.config['UPLOAD_FOLDER'] + "/Threshold/" + dateOfStart + "/")
                    file.save(os.path.join(nextDOM.config['UPLOAD_FOLDER'],"Threshold/" + dateOfStart + "/", filename)) # uloží soubory do teto složky
            filesPath = os.path.join(nextDOM.config['UPLOAD_FOLDER'],"Threshold/" + dateOfStart + "/")
#dolpnit co dělat s CSV00
            return redirect(url_for('graphs', saving=saving, loading=loading, NumberOfNormals=NumberOfNormals,files=filesPath))
    else:
        return render_template("/v2/thresholds.html")
 
@nextDOM.route('/analyze', methods =["GET", "POST"])
def analyze():
    if request.method == "POST":
        saving = request.form.get("saveAs") #načte informace vloženy do fomruláře z HTML stránky
        nameOfRun = request.form['NameOfRun']    
        limits = request.form['Limits']
        part1 = request.files.getlist('part1') #načte všechny soubory vloženy do fomruláře z HTML stránky
        for file in part1:
            filename = secure_filename(file.filename)
            if filename != '':
                folder_path = os.path.join(nextDOM.config['UPLOAD_FOLDER'],"Analyze/"+nameOfRun+"/Part1/")
                os.makedirs(folder_path, exist_ok=True)
                file.save(os.path.join(folder_path, filename)) # uloží soubory do teto složky
        log_path = os.path.join(nextDOM.config['UPLOAD_FOLDER'], "Analyze", nameOfRun + "_app.log")
        logger = logging.getLogger(nameOfRun)
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s')
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        #logging.basicConfig(filename=os.path.join(nextDOM.config['UPLOAD_FOLDER'],"Analyze/" + nameOfRun+ '_app.log'), level=logging.DEBUG, format='%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s')
        part2 = request.files.getlist('part2')  
        for file in part2:
            filename = secure_filename(file.filename)
            if filename != '':
                folder_path = os.path.join(nextDOM.config['UPLOAD_FOLDER'],"Analyze/"+nameOfRun+"/Part2/")
                os.makedirs(folder_path, exist_ok=True)
                file.save(os.path.join(folder_path, filename)) # uloží soubory do teto složky

        partsPath = os.path.join(nextDOM.config['UPLOAD_FOLDER'],"Analyze/"+nameOfRun)
        cesta_out = partsPath+ '/Results/'
        if saving=="1":
            try:
                logger.info('Starting analysis...')
                #a = datetime.datetime.today().strftime('%H:%M:%S')
                #a = datetime.datetime.strptime(a, '%H:%M:%S')
                report_part1_save_xlsx(partsPath, nameOfRun,logger)
                #b = datetime.datetime.today().strftime('%H:%M:%S')
                #b = datetime.datetime.strptime(b, '%H:%M:%S')
                #print("Part1", b - a)
                logger.info('Part 1 report saved.')
                logger.info('Part 2 report starting... Data is normalizing.')
                report_part2_save_xlsx(partsPath, nameOfRun,limits,logger)
                logger.info('Part 2 report saved.')
                if limits==1:
                    logger.info('Finding LoQ limits. Part 3 report starting...')
                else:
                    logger.info('Finding LoD limits. Part 3 report starting...')
                report_part3_save_xlsx(partsPath, nameOfRun,limits,logger)
                logger.info('Part 3 report saved.')
            except Exception as e:
                error_message = str(e)
                return render_template('/v2/error.html', error_message=error_message)
            zip_filename = 'static/uploadedFiles/Analyze_'+nameOfRun+'.zip'
            with zipfile.ZipFile(zip_filename, 'w') as zip_file:
                for root, _, files in os.walk(cesta_out):
                    for file in files:
                        file_path = os.path.join(root, file)
                        zip_file.write(file_path, os.path.relpath(file_path, cesta_out))
            file_handler.close()
            logging.shutdown()
            return send_file(zip_filename, as_attachment=True)
    return render_template("/v2/analyze.html")



@nextDOM.route("/logout") #adresa, která mě odhlásí
def logout():
    session.clear()
    return redirect("/")


def login_is_required(function): # funkce, která ověřuje, zda jsem přihlášen
    def wrapper(*args, **kwargs):
        if bool(session):
            return function()
        else:
            return abort(401)  # Authorization required Vrací hlášku, která říká že nemám práva
    wrapper.__name__ = function.__name__
    return wrapper


@nextDOM.route("/endofwork", methods =["GET", "POST"])
def delete_subfolders():
    try:
        folder_path = nextDOM.config["UPLOAD_FOLDER"]
        current_process = psutil.Process()
        children = current_process.children(recursive=True)
        for child in children:
            child.terminate()
        gone, alive = psutil.wait_procs(children, timeout=5)
        for p in alive:
            p.kill()
        try:
            for root, dirs, files in os.walk(folder_path, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
        except Exception as e:  #Vypsání chybové hlášky na HTML stránce
            error_message = str(e)
            return render_template('/v2/error.html', error_message=error_message)
        try:
            os.makedirs(folder_path+"/Analyze", exist_ok=True)
            os.makedirs(folder_path+"/Threshold", exist_ok=True)
            os.makedirs(folder_path+"/Threshold/Results", exist_ok=True)
        except Exception as e:
            error_message = str(e)
            return render_template('/v2/error.html', error_message=error_message)
    except Exception as e:
        error_message = str(e)
        return render_template('/v2/error.html', error_message=error_message)
    return redirect("/")


@nextDOM.route('/plot')
def graphs():
    saving = request.args.get('saving') # saving=saving, loading=loading, NumberOfNormals=NumberOfNormals)
    loading = request.args.get('loading')
    NumberOfNormals = request.args.get('NumberOfNormals')
    files = request.args.get('files')
    if loading.__contains__("default"):
        komplet_limity_LoB = pd.read_csv('static/defaultCSV/AllSoTLoDComplet.csv', sep="\t")
        limity_LoB = pd.read_csv('static/defaultCSV/AllSoTLoBAll.csv', sep="\t")

        fig = px.scatter(x=komplet_limity_LoB[komplet_limity_LoB.columns[0]], y=komplet_limity_LoB[komplet_limity_LoB.columns[2]] - limity_LoB[limity_LoB.columns[2]])
    # fig.add_trace(px.scatter_3d(x=komplet_limity_LoB.iloc[:, 0], 
    #                          y=-(komplet_limity_LoB.iloc[:, 3] - limity_LoB.iloc[:, 3])))
        fig1=px.scatter(x=komplet_limity_LoB.iloc[:, 0], y=-(komplet_limity_LoB.iloc[:, 3] - limity_LoB.iloc[:, 3]),color_discrete_sequence=['red'])
        fig['data'][0]['showlegend'] = True
        fig['data'][0]['name'] = 'Transition'
        fig1['data'][0]['showlegend'] = True
        fig1['data'][0]['name'] = 'Transversion'
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        graphJSON2 = json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder)
        return render_template('/v2/graphs.html', graphJSON=graphJSON,graphJSON2=graphJSON2,saving=saving)
    elif loading.__contains__("folder"):
        try:
            thresholdResultsPath = matrixCreator_csv(saving,NumberOfNormals,files)
        except Exception as e:
            error_message = str(e)
            return render_template('/v2/error.html', error_message=error_message)
        if saving=="1":
            wb = openpyxl.load_workbook(thresholdResultsPath+'/AllSoTStatistic.xlsx')
            sheet = wb['LoD_komplet']
            sheetLOB = wb['AllSoTLoBAll']
            komplet_limity_LoB = pd.DataFrame(sheet.values, columns=['sloupec1', 'sloupec2', 'sloupec3', 'sloupec4'])
            limity_LoB = pd.DataFrame(sheetLOB.values, columns=['sloupec1', 'sloupec2', 'sloupec3', 'sloupec4'])
        else:
            komplet_limity_LoB = pd.read_csv(thresholdResultsPath+'/AllSoTLoDComplet.csv', sep="\t",header=None)
            limity_LoB = pd.read_csv(thresholdResultsPath+'/AllSoTLoBAll.csv', sep="\t",header=None)
        fig = px.scatter(x=komplet_limity_LoB[komplet_limity_LoB.columns[0]], y=komplet_limity_LoB[komplet_limity_LoB.columns[2]] - limity_LoB[limity_LoB.columns[2]])
    # fig.add_trace(px.scatter_3d(x=komplet_limity_LoB.iloc[:, 0], 
    #                          y=-(komplet_limity_LoB.iloc[:, 3] - limity_LoB.iloc[:, 3])))
        fig1=px.scatter(x=komplet_limity_LoB.iloc[:, 0], y=-(komplet_limity_LoB.iloc[:, 3] - limity_LoB.iloc[:, 3]),color_discrete_sequence=['red'])
        fig['data'][0]['showlegend'] = True
        fig['data'][0]['name'] = 'Transition'
        fig1['data'][0]['showlegend'] = True
        fig1['data'][0]['name'] = 'Transversion'
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        graphJSON2 = json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder)
        return render_template('/v2/graphs.html', graphJSON=graphJSON,graphJSON2=graphJSON2,thresholdResultsPath=thresholdResultsPath,saving=saving)
    else:    
        pass

@nextDOM.route('/treningData', methods=["GET", "POST"])
def treningData():
    zip_path = "static/defaultCSV/parts.zip"
    return send_file(zip_path, as_attachment=True)

@nextDOM.route('/defaultThresholSet', methods=["GET", "POST"])
def defaultThresholSet():
    thresholdResultsPath = request.args.get('thresholdResultsPath')
    newThreshold = thresholdResultsPath+'/SoTNextDOM.csv'
    oldThreshold = "static/defaultCSV/matrice.csv"
    shutil.copyfile(newThreshold, oldThreshold)
    return render_template('/v2/analyze.html')

@nextDOM.route('/downloadDefaultThreshold', methods=["GET", "POST"])
def graph():
    thresholdResultsPath = request.args.get('thresholdResultsPath')
    saving = request.args.get('saving')
    if saving=="0":
        if thresholdResultsPath == '':
            f1 = thresholdResultsPath+'static/defaultCSV/AllSoTLoBAll.csv'
            f2 = thresholdResultsPath+'static/defaultCSV/AllSoTLoDAll.csv'
            f3 = thresholdResultsPath+'static/defaultCSV/AllSoTLoDPos.csv'
            f4 = thresholdResultsPath+'static/defaultCSV/AllSoTLoDComplet.csv'
            f5 = thresholdResultsPath+'static/defaultCSV/AllSoTLoQComplet.csv'
            f6 = thresholdResultsPath+'static/defaultCSV/SoTNextDOM.csv'
        else:
            f1 = thresholdResultsPath+'/AllSoTLoBAll.csv'
            f2 = thresholdResultsPath+'/AllSoTLoDAll.csv'
            f3 = thresholdResultsPath+'/AllSoTLoDPos.csv'
            f4 = thresholdResultsPath+'/AllSoTLoDComplet.csv'
            f5 = thresholdResultsPath+'/AllSoTLoQComplet.csv'
            f6 = thresholdResultsPath+'/SoTNextDOM.csv'
        zip_filename = 'static/uploadedFiles/Thresholds_csv.zip'
        with zipfile.ZipFile(zip_filename, 'w') as zip_file:
            zip_file.write(os.path.join(f1))
            zip_file.write(os.path.join(f2))
            zip_file.write(os.path.join(f3))
            zip_file.write(os.path.join(f4))
            zip_file.write(os.path.join(f5))
            zip_file.write(os.path.join(f6))
        return send_file(zip_filename, as_attachment=True)
    elif saving=="1":
        if thresholdResultsPath == '':
            f1 ='static/defaultCSV/AllSoTStatistic.xlsx'
            f2 ='static/defaultCSV/SoTNextDOM.csv'
        else:
            f1 = thresholdResultsPath+'/AllSoTStatistic.xlsx'
            f2 = thresholdResultsPath+'/SoTNextDOM.csv'
        zip_filename = 'static/uploadedFiles/Thresholds_xlsx.zip'
        with zipfile.ZipFile(zip_filename, 'w') as zip_file:
            zip_file.write(os.path.join(f1))
            zip_file.write(os.path.join(f2))
        return send_file(zip_filename, as_attachment=True)

@nextDOM.route('/gallery')
def examples():
    return render_template('/v2/galery.html')


def my_periodic_function():
    delete_subfolders()



if __name__ == "__main__":   
    scheduler = BackgroundScheduler()
    scheduler.add_job(my_periodic_function, 'interval', minutes=1440)
    scheduler.start()
    nextDOM.run(debug=True,host='0.0.0.0', port=5000) #řádek, který spouští server na adrese 0.0.0.0 -> je viditelný odkudkoli


    