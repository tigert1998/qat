import torch

OPSET = 13
AXIS_OPSET = 11

# The following functions are from:
# https://github.com/Xilinx/brevitas


class QLinearConvFn(torch.autograd.Function):
    @staticmethod
    def symbolic(
            g, int_x,
            input_scale,
            input_zero_point,
            int_weight,
            weight_scale,
            weight_zero_point,
            output_scale,
            ouput_zero_point,
            output_dtype,
            int_bias,
            out_shape,
            kernel_size,
            padding,
            stride,
            groups,
            dilation):
        if int_bias is not None:
            ret = g.op(
                'QLinearConv', int_x,
                input_scale,
                input_zero_point,
                int_weight,
                weight_scale,
                weight_zero_point,
                output_scale,
                ouput_zero_point,
                int_bias,
                kernel_shape_i=kernel_size.tolist(),
                pads_i=padding.tolist(),
                strides_i=stride.tolist(),
                group_i=groups.tolist(),
                dilations_i=dilation.tolist())
        else:
            ret = g.op(
                'QLinearConv', int_x,
                input_scale,
                input_zero_point,
                int_weight,
                weight_scale,
                weight_zero_point,
                output_scale,
                ouput_zero_point,
                kernel_shape_i=kernel_size.tolist(),
                pads_i=padding.tolist(),
                strides_i=stride.tolist(),
                group_i=groups.tolist(),
                dilations_i=dilation.tolist())
        return ret

    @staticmethod
    def forward(
            ctx, int_x,
            input_scale,
            input_zero_point,
            int_weight,
            weight_scale,
            weight_zero_point,
            output_scale,
            output_zero_point,
            output_dtype,
            int_bias,
            out_shape,
            kernel_size,
            padding,
            stride,
            groups,
            dilation):
        return torch.empty(out_shape.tolist(), dtype=output_dtype, device=int_x.device)


class DequantizeLinearFn(torch.autograd.Function):

    @staticmethod
    def symbolic(
            g, x,
            input_scale,
            input_zero_point,
            input_axis):
        if input_axis is not None and OPSET >= AXIS_OPSET:
            ret = g.op(
                'DequantizeLinear', x,
                input_scale,
                input_zero_point,
                axis_i=input_axis)
        else:
            ret = g.op(
                'DequantizeLinear', x,
                input_scale,
                input_zero_point)
        return ret

    @staticmethod
    def forward(
            ctx, int_x,
            input_scale,
            input_zero_point,
            input_axis):
        return int_x.float()


class QuantizeLinearFn(torch.autograd.Function):

    @staticmethod
    def symbolic(
            g, x,
            output_scale,
            ouput_zero_point,
            output_dtype,
            output_axis):
        if output_axis is not None and OPSET >= AXIS_OPSET:
            ret = g.op(
                'QuantizeLinear', x,
                output_scale,
                ouput_zero_point,
                axis_i=output_axis)
        else:
            ret = g.op(
                'QuantizeLinear', x,
                output_scale,
                ouput_zero_point)
        return ret

    @staticmethod
    def forward(
            ctx, x,
            output_scale,
            ouput_zero_point,
            output_dtype,
            output_axis):
        return x.type(output_dtype)


class QLinearMatMulFn(torch.autograd.Function):

    @staticmethod
    def symbolic(
            g, int_x,
            input_scale,
            input_zero_point,
            int_weight,
            weight_scale,
            weight_zero_point,
            output_scale,
            ouput_zero_point,
            output_dtype,
            out_shape):
        ret = g.op(
            'QLinearMatMul', int_x,
            input_scale,
            input_zero_point,
            int_weight,
            weight_scale,
            weight_zero_point,
            output_scale,
            ouput_zero_point)
        return ret

    @staticmethod
    def forward(
            ctx, int_x,
            input_scale,
            input_zero_point,
            int_weight,
            weight_scale,
            weight_zero_point,
            output_scale,
            output_zero_point,
            output_dtype,
            out_shape):
        return torch.empty(out_shape, dtype=output_dtype, device=int_x.device)

# The following functions are written by Xiaohu.


class QLinearLeakyReluFn(torch.autograd.Function):
    @staticmethod
    def symbolic(
            g, int_x,
            input_scale,
            input_zero_point,
            output_scale,
            ouput_zero_point,
            output_dtype,
            alpha):
        return g.op(
            'com.microsoft::QLinearLeakyRelu', int_x,
            input_scale,
            input_zero_point,
            output_scale,
            ouput_zero_point,
            alpha_f=alpha
        )

    @staticmethod
    def forward(
            ctx, int_x,
            input_scale,
            input_zero_point,
            output_scale,
            ouput_zero_point,
            output_dtype,
            alpha):
        return torch.empty_like(int_x).to(output_dtype)


class QLinearAddFn(torch.autograd.Function):
    @staticmethod
    def symbolic(
            g, int_a,
            a_scale,
            a_zero_point,
            int_b,
            b_scale,
            b_zero_point,
            output_scale,
            output_zero_point,
            output_dtype):
        return g.op(
            'com.microsoft::QLinearAdd',
            int_a,
            a_scale,
            a_zero_point,
            int_b,
            b_scale,
            b_zero_point,
            output_scale,
            output_zero_point
        )

    @staticmethod
    def forward(
            ctx, int_a,
            a_scale,
            a_zero_point,
            int_b,
            b_scale,
            b_zero_point,
            output_scale,
            output_zero_point,
            output_dtype):
        return torch.empty_like(int_a + int_b).to(output_dtype)


class QLinearGlobalAveragePoolFn(torch.autograd.Function):
    @staticmethod
    def symbolic(
            g, int_x,
            input_scale,
            input_zero_point,
            output_scale,
            output_zero_point,
            output_dtype,
            out_shape,
            channels_last):
        return g.op(
            'com.microsoft::QLinearGlobalAveragePool',
            int_x,
            input_scale,
            input_zero_point,
            output_scale,
            output_zero_point,
            channels_last_i=channels_last
        )

    @staticmethod
    def forward(
            ctx, int_x,
            input_scale,
            input_zero_point,
            output_scale,
            output_zero_point,
            output_dtype,
            out_shape,
            channels_last):
        return torch.empty(out_shape.tolist(), dtype=output_dtype)


class QGemmFn(torch.autograd.Function):
    @staticmethod
    def symbolic(
            g, int_a,
            a_scale,
            a_zero_point,
            int_b,
            b_scale,
            b_zero_point,
            int_c,
            output_scale,
            output_zero_point,
            output_dtype,
            out_shape,
            trans_a,
            trans_b,
            alpha):
        return g.op(
            'com.microsoft::QGemm',
            int_a,
            a_scale,
            a_zero_point,
            int_b,
            b_scale,
            b_zero_point,
            int_c,
            output_scale,
            output_zero_point,
            transA_i=trans_a,
            transB_i=trans_b,
            alpha_f=alpha
        )

    @staticmethod
    def forward(
            ctx, int_a,
            a_scale,
            a_zero_point,
            int_b,
            b_scale,
            b_zero_point,
            int_c,
            output_scale,
            output_zero_point,
            output_dtype,
            out_shape,
            trans_a,
            trans_b,
            alpha):
        return torch.empty(out_shape.tolist(), dtype=output_dtype)


class MaxPoolFn(torch.autograd.Function):
    @staticmethod
    def symbolic(
            g, x,
            out_shape,
            output_dtype,
            kernel_shape,
            pads,
            strides):
        return g.op(
            'MaxPool',
            x,
            kernel_shape_i=kernel_shape.tolist(),
            pads_i=pads.tolist(),
            strides_i=strides.tolist()
        )

    @staticmethod
    def forward(
            g, x,
            out_shape,
            output_dtype,
            kernel_shape,
            pads,
            strides):
        return torch.empty(out_shape.tolist(), dtype=output_dtype)
