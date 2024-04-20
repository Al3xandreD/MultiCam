package com.example.final_app;

import android.os.Parcel;
import android.os.Parcelable;

public class DataEtage implements Parcelable {
    public String nom, camera, victime;
    public int logo_carte, logo_camera;

    public DataEtage(String nom, String camera, String victime, int logo_carte, int logo_camera) {
        this.nom = nom;
        this.camera = camera;
        this.victime = victime;
        this.logo_carte = logo_carte;
        this.logo_camera = logo_camera;
    }

    // Parcelable methods

    // Constructor for parcelable
    protected DataEtage(Parcel in){
        nom = in.readString();
        camera = in.readString();
        victime = in.readString();
        logo_carte = in.readInt();
        logo_camera = in.readInt();
    }

    @Override
    public void writeToParcel(Parcel dest, int flags) {
        dest.writeString(nom);
        dest.writeString(camera);
        dest.writeString(victime);
        dest.writeInt(logo_carte);
        dest.writeInt(logo_camera);
    }

    // Creator constant for Parcelable
    public static final Creator<DataEtage> CREATOR = new Creator<DataEtage>() {
        @Override
        public DataEtage createFromParcel(Parcel in) {
            return new DataEtage(in);
        }

        @Override
        public DataEtage[] newArray(int size) {
            return new DataEtage[size];
        }
    };

    // Describe the kinds of special objects contained in this Parcelable instance.
    @Override
    public int describeContents() {
        return 0;
    }

    ////////////////////////////getter and setter //////////////////////////

    public String getNom() {
        return nom;
    }

    public void setNom(String nom) {
        this.nom = nom;
    }

    public String getCamera() {
        return camera;
    }

    public void setCamera(String camera) {
        this.camera = camera;
    }

    public String getVictime() {
        return victime;
    }

    public void setVictime(String victime) {
        this.victime = victime;
    }

    public int getLogo_carte() {
        return logo_carte;
    }

    public void setLogo_carte(int logo_carte) {
        this.logo_carte = logo_carte;
    }

    public int getLogo_camera() {
        return logo_camera;
    }

    public void setLogo_camera(int logo_camera) {
        this.logo_camera = logo_camera;
    }
}
