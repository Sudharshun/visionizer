# Visionizer

Tell a great story with Visionizer! Visionizer is a web application that generates a story from a given image. It uses the OpenAI API to generate vision mission and roadmap for your projects

## Setup

1. If you don’t have Python installed, [install it from here](https://www.python.org/downloads/)

2. Clone this repository

3. Navigate into the project directory

   ```bash
   $ cd visionizer
   ```

4. Create a new virtual environment

   ```bash
   # Linux
   $ python -m venv venv
   $ . venv/bin/activate
   ```

   ```shell
   # Windows
   python -m venv venv
   venv\Scripts\activate
   ```

5. Install the requirements

   ```bash
   $ pip install -r requirements.txt
   ```

6. Make a copy of the example environment variables file

   ```bash
   # Linux
   $ cp .env.example .env
   ```

   ```shell
   # Windows
   xcopy .env.example .env
   ```

7. Add your [API key](https://beta.openai.com/account/api-keys) to the newly created `.env` file

8. Run the app

   ```bash
   $ python3 app.py
   ```
You should now be able to access the app at [http://localhost:5001](http://localhost:5001)! 

   ![demo](https://user-images.githubusercontent.com/59533593/173504130-6b36bad6-267a-45b2-96b9-14abe9493ad1.gif)

