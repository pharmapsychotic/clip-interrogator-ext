# CLIP Interrogator extension for Stable Diffusion WebUI

This extension adds a tab for [CLIP Interrogator](https://github.com/pharmapsychotic/clip-interrogator)



![Interrogator tab screenshot](https://github.com/pharmapsychotic/clip-interrogator-ext/raw/main/images/prompt_tab.png)


## Installing

* Go to extensions tab
* Click "Install from URL" sub tab
* Paste `https://github.com/pharmapsychotic/clip-interrogator-ext` and click Install
* Check in your terminal window if there are any errors (if so let me know!)
* Restart the Web UI and you should see a new **Interrogator** tab


## API

The CLIP Interrogator exposes a simple API to interact with the extension which is 
documented on the /docs page under /interrogator/* (using --api flag when starting the Web UI)
* /interrogator/models
  * lists all available models for interrogation
* /interrogator/prompt
  * returns a prompt for the given image, model and mode
* /interrogator/analyze
  * returns a list of words and their scores for the given image, model