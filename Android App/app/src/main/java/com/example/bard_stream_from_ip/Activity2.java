package com.example.bard_stream_from_ip;

import android.app.Activity;
import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;

import androidx.appcompat.app.AppCompatActivity;

public class Activity2 extends AppCompatActivity {
    private EditText ip;
    private Button enter_ip;
    public static String ip_address;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_2);

        ip = (EditText)findViewById(R.id.edit_ip_address);
        enter_ip = (Button) findViewById(R.id.button_ip);

        enter_ip.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                ip_address =ip.getText().toString();
                Intent intent = new Intent(Activity2.this, MainActivity.class);
                startActivity(intent);
            }
        });
    }
}