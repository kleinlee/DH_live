# version 330
precision mediump float;
in mediump vec2 v_texture;
in mediump vec2 v_bias;
out highp vec4 out_color;

void main()
{
    if (v_texture.x == 2.0f)
    {
        out_color = vec4(1.0, 0.0, 0.0, 1.0);
    }
    else if (v_texture.x > 2.0f && v_texture.x < 2.1f)
    {
        out_color = vec4(0.5f, 0.0, 0.0, 1.0);
    }
    else if (v_texture.x == 3.0f)
    {
        out_color = vec4(0.0, 1.0, 0.0, 1.0);
    }
    else if (v_texture.x == 4.0f)
    {
        out_color = vec4(0.0, 0.0, 1.0, 1.0);
    }
    else if (v_texture.x > 3.0f && v_texture.x < 4.0f)
    {
        out_color = vec4(0.0, 0.0, 0.0, 1.0);
    }
    else
    {
        vec2 wrap = (v_bias.xy + 1.0)/2.0;
        out_color = vec4(wrap.xy, 0.5, 1.0);
    }
}