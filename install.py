import launch

if not launch.is_installed("clip-interrogator"):
    launch.run_pip("install clip-interrogator==0.5.1", "requirements for CLIP Interrogator")
