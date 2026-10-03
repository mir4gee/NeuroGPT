`timescale 1ns / 1ps
// Loads x.hex (11,000 int8), runs one chunk, compares all 476*40 int32 results with expected.hex.
module tb_frontend_accel;
    reg clk = 0, rst = 1, start = 0, x_we = 0;
    reg [13:0] x_addr = 0;
    reg signed [7:0] x_din = 0;
    reg [14:0] y_addr = 0;
    wire signed [31:0] y_dout;
    wire done;

    frontend_accel dut (.clk(clk), .rst(rst), .start(start), .x_we(x_we), .x_addr(x_addr), .x_din(x_din),
                        .y_addr(y_addr), .y_dout(y_dout), .done(done));
    always #5 clk = ~clk;  // 100 MHz

    reg [7:0]  xin [0:10999];
    reg [31:0] expv [0:19039];
    integer i, errors, cycles;

    initial begin
        $readmemh("x.hex", xin);
        $readmemh("expected.hex", expv);
        repeat (4) @(posedge clk);
        rst = 0;
        for (i = 0; i < 11000; i = i + 1) begin
            @(negedge clk); x_we = 1; x_addr = i; x_din = xin[i];
        end
        @(negedge clk); x_we = 0;
        @(negedge clk); start = 1;
        @(negedge clk); start = 0;
        cycles = 1;
        while (!done) begin @(posedge clk); cycles = cycles + 1; end
        errors = 0;
        for (i = 0; i < 19040; i = i + 1) begin
            @(negedge clk); y_addr = i;
            @(negedge clk);
            if (y_dout !== $signed(expv[i])) begin
                if (errors < 5) $display("mismatch at %0d: got %0d expected %0d", i, y_dout, $signed(expv[i]));
                errors = errors + 1;
            end
        end
        $display("RESULT outputs=19040 mismatches=%0d cycles_per_chunk=%0d", errors, cycles);
        $finish;
    end
endmodule
