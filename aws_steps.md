## Step 1 - Quick Deploy
./quick_deploy.sh \
  -h 13.234.204.208 \
  -u ec2-user \
  -i "/Users/arun/NPlusLabs/Voice Cloner/Test1.pem" \
  -d /home/ec2-user/drivebench

## Step 2 - SSH into aws instance
ssh -i "/Users/arun/NPlusLabs/Voice Cloner/Test1.pem" ec2-user@13.234.204.208

## Step 3 - Install git
sudo dnf install git -y