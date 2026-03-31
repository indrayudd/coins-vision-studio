# Project 1 - Midterm

## AI/External Sources
- Google Search and ChatGPT used to understand CV concepts.
- ChatGPT used for the frontend.
- ChatGPT prompts:
  - Implement the UI for backend/ as written in the documentation Project 1 Proposal.pdf. Make the UI react based and make it the most beautiful like a designer's portfolio. There should be a 'draw' button that draws from the dataset and shows the step by step transformation as written in the spec.
  - there is no visual distinction between global stats and drawn stats. not good ux
  - there is very little distinciton between thumbnails of the drawn images and steps of the chosen image. i don't need thumbnails for the chosen image, and the information is cluttered. i don't need "error" -1. -8. etc on the thumbnail itself when slected counts already shows that in the panel.
  - The right panel looks empty. If you could add more visualization about the nitty gritties of the transform or how the substeps of the transform work (as shown in pasted example), that would be a lot more informative and well rounded as a project.
  - the panel needs to be better. understand that this is for non technical, non cs folks who have no idea, so you have to simplify the jargon, break the transformations down to its core matrix mathematics, and visualize the transformation steps in the most beautiful and innovative way out there so that the reader has no doubt in their mind. you can search the web for best practices to viz these sub transformations.

- https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html
- https://docs.opencv.org/4.x/de/dbc/tutorial_py_fourier_transform.html
- https://docs.opencv.org/4.x/dc/dd3/tutorial_gausian_median_blur_bilateral_filter.html
- https://docs.opencv.org/3.4/d9/db0/tutorial_hough_lines.html
- https://docs.opencv.org/4.x/da/d53/tutorial_py_houghcircles.html
- https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html
- https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html
- https://docs.opencv.org/3.4/d3/dc0/group__imgproc__shape.html
- https://docs.opencv.org/3.4/d4/d73/tutorial_py_contours_begin.html
- https://docs.opencv.org/3.4/d2/dbd/tutorial_distance_transform.html
- https://docs.opencv.org/3.4/df/d3d/tutorial_py_inpainting.html
- https://scikit-image.org/docs/0.25.x/auto_examples/features_detection/plot_blob.html

## How to Run
### Prerequisites

- **Git** (to clone the repo): https://git-scm.com/downloads
- **Node.js (LTS)** (includes `npm`): https://nodejs.org/en/download
- **Python 3.10+**: https://www.python.org/downloads/


These must run before getting started. If they don't revisit the previous steps:
```bash
git --version
node --version
npm --version
python --version
```

### 1) Clone the repository
```bash
git clone https://github.com/indrayudd/coins-vision-studio.git
cd coins-vision-studio
```

### 2) Install frontend dependencies
```bash
npm install
```

### 3) Install backend Python dependencies
Use your active Python environment:
```bash
python -m pip install -r requirements.txt
```

### 4) Start backend (Terminal/CMD 1)
```bash
npm run dev:api
```

If `npm run dev:api` uses the wrong Python interpreter on your machine, run backend directly:
```bash
python -m uvicorn backend.main:app --reload --port 8000
```

### 5) Start frontend (Terminal/CMD 2)
```bash
npm run dev:web
```

### 6) Open the app
- Frontend: http://127.0.0.1:5173
- Backend health: http://127.0.0.1:8000/api/health

### FAQs
-  How to open the terminal?
  - Windows: Press the start button and search 'CMD'
  - MacOS: CMD+Space -> Search 'terminal'
  - Linux: You know if you are on linux

## Notes:
The first start will take some time because the algortihms run on all 87 images. Be patient! It's not a bug.