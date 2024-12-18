def log(
    logger, step=None, losses=None, fig=None, audio=None, sampling_rate=22050, tag=""
):
    if losses is not None:
        logger.add_scalar("Loss/d_loss", losses[0], step)
        logger.add_scalar("Loss/g_gan_loss", losses[1], step)
        logger.add_scalar("Loss/g_l1_loss", losses[2], step)

    if fig is not None:
        logger.add_image(tag, fig, 2, dataformats='HWC')

    if audio is not None:
        logger.add_audio(
            tag,
            audio / max(abs(audio)),
            sample_rate=sampling_rate,
        )