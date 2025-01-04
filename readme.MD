# MicroClassify Website 

## Project Structure

```sh
Directory structure:
└── Incharajayaram-radarvision-webpage/
    ├── deployed_website/
    │   ├── README.md
    │   ├── Dockerfile
    │   ├── LICENSE
    │   ├── app.py
    │   ├── config.py
    │   ├── docker-compose.yaml
    │   ├── requirements.txt
    │   ├── .gitignore
    │   ├── static/
    │   │   ├── assets/
    │   │   ├── images/
    │   │   ├── sample_images/
    │   │   └── scripts/
    │   │       └── script.js
    │   ├── templates/
    │   │   ├── contact.css
    │   │   ├── contact.html
    │   │   ├── landingpage.html
    │   │   ├── login.html
    │   │   ├── signup.html
    │   │   └── upload.html
    │   └── uploads/
    └── rpi_website/
        ├── app.py
        ├── appwithatt.py
        ├── shuffled_images/
        ├── static/
        │   └── style.css
        └── templates/
            └── index.html
```

## Folder Content

```sh
deployed_website: contains the code for frontend and backend of the custom website made for MicroClassify as a product.
rpi_website: website to be deployed on raspberry PI 5, and it has a streaming pipeline to provide model inference in real time classification.
```