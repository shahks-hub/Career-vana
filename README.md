<!-- TABLE OF CONTENTS -->
<h2 id="table-of-contents"> :book: Table of Contents</h2>

<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about-the-project"> About The Project</a></li>
    <li><a href="#overview"> Overview</a></li>
    <li><a href="#getting-started"> Getting Started</a></li>
    <li><a href="#setup"> What We Used</a></li>
  </ol>
</details>

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<!-- ABOUT THE PROJECT -->
<h2 id="about-the-project"> :pencil: About The Project</h2>

<p align="justify"> 
Unlock the full potential of your tech prowess with our revolutionary application! Navigating the diverse landscape of the tech sector can be daunting, but fear not—our app is your key to decoding the perfect career path. Seamlessly translating your skills into a spectrum of lucrative roles, we empower you to discover the most coveted positions aligned with your expertise.
</p>
<p align="justify">
But that's not all—we go above and beyond by transforming your resume into a powerful weapon. With a simple upload, our app crafts personalized cover letters for each job posting, sparing you the tedious task of manual customization. Elevate your job search game and embark on a tech career journey like never before!
</p>

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<!-- OVERVIEW -->
<h2 id="overview"> :cloud: Overview</h2>

<p align="justify"> 
  In order to use our app, one simply enters the tech skills they feel proficient with, or are simply interested in, into the <code>Enter Your Tech Skills</code> text box on the <code>Sector Selector</code> tab. This will output valuable information regarding the skill sets given, as well as some job postings for those skill sets
</p>

<p align="justify"> 
  Next, simply take one of the provided postings and input it into the <code>Job Description</code> text box on the <code>Generate Cover Letter</code> tab along with a .pdf of their resume into the <code>Provide a .PDF of Your Resume</code> box.
</p>

<p align="justify"> 
  Enjoy your custom Cover Letter, and feel free to use this for any outside job postings you may be interested in!
</p>

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<!-- GETTING STARTED -->
<h2 id="getting-started"> :book: Getting Started</h2>

<p>Create an Environment:</p>
<pre><code>$ python3 -m venv myenv</code></pre>

<p>Activate the Environment:</p>
<pre><code>$ source myenv/bin/activate</code></pre>

<p>Deactivate Environments:</p>
<pre><code>$ deactivate</code></pre>

<p>Install All Dependencies:</p>
<pre><code>$ pip install -r requirements.txt</code></pre>

<p>Run the App:</p>
<pre><code>$ streamlit run main.py</code></pre>

<p>Add Packages to Requirements.txt File:</p>
<pre><code>$ pip freeze > requirements.txt</code></pre>

<i>Note that all of the commands that appear in this project also appear in <code>commands.txt</code>, for easy copying and pasting.</i>

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<!-- SETUP -->
<h2 id="setup"> :floppy_disk: What We Used</h2>

<ul>
  <li><b><a href="https://huggingface.co/">Hugging Face</a></b> - Provided the model for our Cover Letter Generator</li>
  <li><b><a href="https://streamlit.io//">Streamlit</b> - We coded our app on Streamlit and every part of it exists within a Streamlit framework</li>
  <li><b><a href="https://jupyter.org/">Jupyter Notebook</b> - We used Jupyter Notebook to model our data so that we could plug the results into Streamlit.</li>
  <li><b><a href="https://pandas.pydata.org/">Pandas</b> - We would not be able to model our data without the Pandas library.</li>
  <li><b><a href="https://numpy.org/">Numpy</b> - Every bit as crucial to our data modeling as Pandas.</li>
  <li><b><a href="https://scikit-learn.org/stable/">Scikit-Learn</b> - </li>
  <li><b><a href="https://openai.com/">OpenAI</b> - </li>
</ul>

<h3>Some of our most valuable references</h3>
<ul>
  <li><b><a href="https://plotly.com/python/plotly-express/">Plotly</b> - A great reference point for several types of visual models</li>
  <li><b><a href="https://chat.openai.com/">ChatGPT</b> - Every time we got stuck on a bit of code, ChatGPT was there to help us out.</li>
  <li><b><a href="https://docs.streamlit.io/library/advanced-features/theming">Streamlit Themeing</b> - Valuable for helping to adjust the theme of your streamlit app</li>
  <li><b><a href="https://www.kaggle.com/">Kaggle</b> - Where we went to find the datasets for our visuals and to train our models.</li>
</ul>

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)
