import os
import sys
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify, send_file
from werkzeug.utils import secure_filename
import json
import random
import string
import datetime
import endpoints

import logger
"""
        Flask Server for API
"""
# Init the flask application
app = Flask(__name__)

UPLOAD_DIR = './data/temp/'
ALLOW_EXT = set(['json', 'txt'])


def allow_ext(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOW_EXT


@app.route('/')
def prog():
    return render_template('prog.html')


logo_mark_str = """\n
        _  _   _   __
       /||/|| //  ||||))
      //|//||//__ ||||
\n"""


@app.route('/submit', methods=['POST'])
def submit():
    # Support uploading files, just in case
    if len(request.files) > 0:
        file = request.files['file']
        json_fn = secure_filename(file.filename)

        logger.log_print("{}\n".format(request))

        if file and allow_ext(file.filename):
            try:
                logger.log_print(logo_mark_str)

                logger.log_print("uplaoding...\n")
                # check if there is uploader folder is exist
                if not os.path.isdir(UPLOAD_DIR):
                    os.mkdir(UPLOAD_DIR)
                # remove all preworked json files in uploader directory
                for fname in os.listdir(UPLOAD_DIR):
                    if os.path.isfile(os.path.join(fname, UPLOAD_DIR)):
                        os.remove(os.path.join(UPLOAD_DIR, fname))

                # save json file on Upload_dir
                file.save(os.path.join(UPLOAD_DIR, json_fn))
                logger.log_print("Save the json file to filename{}\n".format(json_fn))

                # start the progress
                logger.log_print("Start the Progress\n")
                result_dict = endpoints.progress(UPLOAD_DIR, json_fn)
                logger.log_print("End the Progress\n")

                # return the result dict as a json file
                # now = datetime.datetime.now()
                # result_fn = "result_{}_{:02d}-{:02d}-{:02d}_{}".format(str(now.date()), now.hour, now.minute, now.second,
                #                                                        json_fn)
                # result_path = os.path.join(UPLOAD_DIR, result_fn)
                # with open(result_path, 'w') as fp:
                #     json.dump(result_dict, fp)
                #
                # return send_file(result_path, attachment_filename=result_fn)
                return jsonify(results=result_dict)

            except Exception as e:
                str = 'Exception : {}'.format(e)
                logger.log_print(str + "\n")
                return str
        else:
            str = 'No allowed file format'.format(json_fn)
            logger.log_print(str + "\n")
            return str

    # Preferred, post json data
    elif len(request.form) > 0:
        try:
            item = request.form.items()[0]
            if item[1] == '':
                data = item[0]
            else:
                data = item[1]

            # datadict = json.loads(data)

            # Just generate random name
            rand_str = lambda n: ''.join([random.choice(string.lowercase) for i in xrange(n)])
            s = rand_str(10) + ".json"
            json_fn = secure_filename(s)
            
            with open(os.path.join(UPLOAD_DIR, json_fn), 'w') as file:
                # json.dump(datadict, file)
                file.write(data)
                file.close()

            logger.log_print("Start the Progress\n")
            result_dict = endpoints.progress(UPLOAD_DIR, json_fn)
            logger.log_print("End the Progress\n")

            return jsonify(results=result_dict)

        except Exception as e:
            str = 'Exception : {}'.format(e)
            logger.log_print(str + "\n")
            return str
    else:
        str = 'No file or post data found'
        logger.log_print(str + "\n")
        return str

if __name__ == '__main__':

    port = int(os.environ.get('PORT', 5000))

    app.run(
        host='0.0.0.0',
        port=port,
        debug=True,
        threaded=True
    )
