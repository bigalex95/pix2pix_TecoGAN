# Video Chat with WebRTC and Firebase

Build a 1-to-1 video chat feature with WebRTC, Firestore, and JavaScript. 

Watch the [WebRTC Explanation on YouTube](https://youtu.be/WmR9IMUD_CY) and follow the full [WebRTC Firebase Tutorial](https://fireship.io/lessons/webrtc-firebase-video-chat) on Fireship.io. 


## Usage

### Step 1. Clone repository
```
git clone https://github.com/bigalex95/pix2pix_TecoGAN
cd ./webrtc-demo
```
### Step 2. Register to [Google Firebase](https://firebase.google.com/) and create test-mode database. After creating project and database, update the firebase project config in the main.js file.
```
const firebaseConfig = {
    //enter config here
};
```
### Step 3. (Optional) Add necessary configurations (port, server type and etc.) to vite.config.js file.
Example of how change config file:
if you want change
```
server.port = Type: number
```
you need write this like below:
```
server: {
    port: 80, //number
  },
```
### Step 4. Install necessary packages to running vitejs server.
```
npm install
```
### Step 5. Run vitejs server.
```
npm run dev
```