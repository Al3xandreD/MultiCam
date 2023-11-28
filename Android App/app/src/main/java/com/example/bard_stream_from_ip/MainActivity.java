package com.example.bard_stream_from_ip;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Context;
import android.graphics.drawable.BitmapDrawable;
import android.graphics.drawable.Drawable;
import android.media.MediaPlayer;
import android.os.Bundle;

import android.widget.TextView; // pour affichage de la saisie au clavier

import android.os.Handler;
import android.widget.ImageView;
import android.widget.Toast;
import android.os.Bundle;
import com.bumptech.glide.Glide;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.AsyncTask;
import android.os.Bundle;
//import android.support.v7.app.AppCompatActivity;
import android.widget.ImageView;

import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.URL;

import static com.example.bard_stream_from_ip.Activity2.ip_address;


public class MainActivity extends AppCompatActivity {

    private static final long FRAME_RATE = 60; // 60 frames per second (adjust as needed)
    private Handler handler;
    private ImageView imageView;
    long frameDelay = 1000/FRAME_RATE;
    // Set the video URL
    String[] videoUrl = {"https://www.google.com/imgres?imgurl=https%3A%2F%2Fpng.pngtree.com%2Felement_our%2F20190603%2Fourmid%2Fpngtree-black-blue-butterfly-cartoon-illustration-image_1446248.jpg&tbnid=ynJaRhaZI9KkfM&vet=12ahUKEwjy4_GciOeCAxUXTaQEHRUkBcQQMygDegQIARBZ..i&imgrefurl=https%3A%2F%2Ffr.pngtree.com%2Ffree-animals-png%2Fbutterfly&docid=v8SrOHk6DLpgfM&w=360&h=360&q=papillon%20jpg&ved=2ahUKEwjy4_GciOeCAxUXTaQEHRUkBcQQMygDegQIARBZ.jpg"}; //{"http://192.168.43.190/cam-lo.jpg"};
    //String[] videoUrl1 = {"http://" + ip_address + "/cam-lo.jpg"}; //test avec seulement un String pour faciliter la concaténation

    HttpURLConnection connection;

    private boolean canBeRelaunch_flag = true;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // affichage de la saisie au clavier
        TextView textView = findViewById(R.id.textViewId);
        String contenuVariable = ip_address;
        // Définissez le texte du TextView avec le contenu de votre variable
        textView.setText(contenuVariable);

        //******************* UI Initialization *******************
        imageView = findViewById(R.id.imageView);


        //************* Retrieve URLs *************




        //******************* Main Loop *******************
        handler = new Handler();
        Update();
    }


    private void Update() {
        handler.postDelayed(new Runnable() {
            @Override
            public void run() {

                //Update frame
                new DownloadImageTask().execute(videoUrl);

                // Call the next iteration after the specified frame rate
                handler.postDelayed(this, frameDelay);
            }
        }, frameDelay);
    }

    private class DownloadImageTask extends AsyncTask<String, Void, Bitmap> {

        @Override
        protected Bitmap doInBackground(String... urls) {
            String imageUrl = urls[0];
            Bitmap bitmap = null;

            try {
                URL url = new URL(imageUrl);
                HttpURLConnection connection = (HttpURLConnection) url.openConnection();
                connection.setDoInput(true);
                connection.connect();
                InputStream input = connection.getInputStream();
                bitmap = BitmapFactory.decodeStream(input);
            } catch (Exception e) {
                e.printStackTrace();
            }
            return bitmap;
        }

        @Override
        protected void onPostExecute(Bitmap result) {
            if (result != null) {
                // Set the downloaded image to the ImageView
                imageView.setImageBitmap(result);
            }
        }
    }


    @Override
    protected void onDestroy() {
        super.onDestroy();
    }
}