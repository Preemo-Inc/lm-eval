# Getting started

# 1. Installations

There is a quick setup for Linux:
```bash
. ./bin/dev/setup.sh
```

### 1.2. Manual install
If you cannot follow the script above, install the following:

```bash
export SKYPILOT_DISABLE_USAGE_COLLECTION=1
pip install "skypilot[aws]"
```

For AWS: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html
```bash
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
```

# 2. Adding Cloud Providers:
Follow the https://skypilot.readthedocs.io/en/latest/cloud-setup/cloud-auth.html.

## AWS
### Get your secret key 
Go to the console, get the Key. Afaik best way to get it from Identity and Access Management (IAMv2)
https://us-east-1.console.aws.amazon.com/iamv2/home?region=us-east-1#/users

You can also directly go to your user in our AWS, download secret key in the top right in `create access key`
There is also an experimental `skypilot-v1` user, that has too narrow restrictions. We might discontinue it.

### Set up the client with the secret key
You basically need to follow https://skypilot.readthedocs.io/en/latest/cloud-setup/cloud-auth.html.
For me ```aws configure sso``` did not work, with our subscription hence I used the secret key.

Enter
```bash
# AWS configure, have the downloaded csv ready.
aws configure list
```
# 3. Launch a Job

#### Launch a job sky_v100_aws.yaml
More CLI argmuents: https://skypilot.readthedocs.io/en/latest/reference/cli.html
```bash
cd gradient_infra/
sky launch --env-file .env -c sky-lm-eval-gpu sky_v100_aws.yaml
```

#### Terminate a cluster:
```bash
sky stop sky-lm-eval-gpu
```

#### Experience:
Launching a spot_instance job with `--down` is a good option to delete all resource directly after usage
Azure has not the best user experience, `az login` is annoying. AWS has better feeling with skypilot.
