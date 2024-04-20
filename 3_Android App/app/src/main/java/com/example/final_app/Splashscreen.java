package com.example.final_app;

import android.content.Intent;
import android.graphics.Color;
import android.os.Build;
import android.os.Bundle;
import android.view.MotionEvent;
import android.view.View;
import android.view.Window;
import android.view.WindowManager;

import com.example.final_app.databinding.ActivitySplashscreenBinding;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

public class Splashscreen extends AppCompatActivity {

    private ActivitySplashscreenBinding binding;

    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        binding = ActivitySplashscreenBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        binding.getRoot().setOnTouchListener(new View.OnTouchListener(){
            @Override
            public boolean onTouch(View v , MotionEvent event) {
                //check for touch
                if ( event.getAction() == MotionEvent.ACTION_DOWN){
                    // Start the new activity on touch
                    Intent intent = new Intent(Splashscreen.this, MainActivity.class);
                    startActivity(intent);
                    // Finish the current activity to prevent going back
                    finish();
                }
                return true;
            }
        });

    }
}
